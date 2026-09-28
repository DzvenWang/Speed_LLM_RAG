#!/usr/bin/env python3
from __future__ import annotations

import argparse, copy, json, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

GADI_ROOT = next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd())
ROOT = GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.runners.common import make_run_dir, masked_mse, require_gpu, save_val_outputs, scale_stats, set_seed, validate_stage


class MoGERNNDataset(Dataset):
    def __init__(self, cache: Path, split: str):
        self.x = np.load(cache/f'{split}_x_obs_norm.npy', mmap_mode='r')
        self.obs = np.load(cache/f'{split}_history_observed.npy', mmap_mode='r')
        self.y = np.load(cache/f'{split}_y_norm.npy', mmap_mode='r')
        self.valid = np.load(cache/f'{split}_future_valid.npy', mmap_mode='r')
    def __len__(self): return int(self.x.shape[0])
    def __getitem__(self, i):
        return (np.array(self.x[i], np.float32, copy=True), np.array(self.obs[i], np.uint8, copy=True),
                np.array(self.y[i], np.float32, copy=True), np.array(self.valid[i], np.uint8, copy=True))


def import_model():
    repo = ROOT/'vendor/MoGERNN'
    if str(repo) not in sys.path: sys.path.insert(0, str(repo))
    try:
        from model.MoGERNN import Model
    except Exception as e:
        raise RuntimeError(f'Cannot import official MoGERNN Model from {repo}: {e!r}') from e
    return Model


def model_config(cfg: dict) -> SimpleNamespace:
    c=cfg['runners']['MoGERNN']; H=int(cfg['experiment']['history']); P=int(cfg['experiment']['horizon'])
    experts=set(str(x) for x in c.get('expert_list',['Mean','WMean','Max','Min','Diffusion']))
    flags={
        'mean_expert':'Mean' in experts, 'weight_expert':'WMean' in experts,
        'max_expert':'Max' in experts, 'min_expert':'Min' in experts,
        'diffusion_expert':'Diffusion' in experts,
    }
    return SimpleNamespace(
        input_len=H, pred_len=P, look_back=0, d_model=int(c['d_model']), k_hop=int(c['k_hop']),
        activation='gelu', e_layers=int(c['e_layers']), d_layers=1, num_used_experts=int(c['num_used_experts']),
        enc_in=1, c_out=1, enc_mark_in=0, t_mark=False, pos_mark=False,
        num_experts=sum(int(v) for v in flags.values()), **flags,
    )


def build_model(cfg: dict, device: torch.device):
    M=import_model(); m=M(model_config(cfg)).to(device)
    return m


def _prepare_x(x: torch.Tensor, obs: torch.Tensor, eps: float) -> torch.Tensor:
    # Official MoGERNN uses numeric zero as its missing sentinel. Protect an
    # actually observed normalized value of exactly zero so it is not reclassified.
    x=x.clone(); obs=obs.bool()
    observed_zero=obs & (x==0)
    x[observed_zero]=float(eps)
    x[~obs]=0.0
    return x.unsqueeze(-1)


def forward_official(model, adj, x, y, epoch: int, device: torch.device):
    # MoGERNN.py creates torch.eye(...) without a device argument. A device context
    # keeps that factory on the same GPU as adj while leaving the official source unchanged.
    with torch.device(device):
        return model(adj, x, None, None, y, int(epoch))


def assert_eval_target_independence(model, adj, x, y, epoch: int, device: torch.device, atol: float) -> dict:
    """Check that eval output is independent of the true future target y.

    If this fails we abort; we do not invent a replacement decoder input in the
    baseline adapter. The pinned vendor inference path must then be reproduced.
    """
    was_training=model.training
    model.eval()
    alternatives={'zeros':torch.zeros_like(y),'time_reversed':torch.flip(y,dims=[1])}
    with torch.no_grad():
        ref=forward_official(model,adj,x,y,epoch,device).detach()
        diffs={}
        for name,alt in alternatives.items():
            pred=forward_official(model,adj,x,alt,epoch,device).detach()
            diffs[name]=float((ref-pred).abs().max().item())
    model.train(was_training)
    result={'epoch':int(epoch),'atol':float(atol),'max_abs_diff':diffs}
    if any(v > float(atol) for v in diffs.values()):
        raise RuntimeError(
            'MoGERNN eval output depends on future target y under the current adapter: '
            + json.dumps(result)
            + '. Do not use this result; reproduce the pinned vendor inference path instead.'
        )
    print('MOGERNN EVAL TARGET-INDEPENDENCE PASS '+json.dumps(result),flush=True)
    return result


def predict(model, dl, adj, device, eps: float, mean: float, std: float, epoch: int) -> np.ndarray:
    model.eval(); out=[]
    with torch.no_grad():
        for x,obs,y,_valid in dl:
            x=_prepare_x(x.float().to(device),obs.to(device),eps)
            y=y.float().to(device).unsqueeze(-1)
            p=forward_official(model,adj,x,y,epoch,device).squeeze(-1)
            out.append((p.cpu().numpy()*np.float32(std)+np.float32(mean)).astype(np.float32))
    return np.concatenate(out,0)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['development','formal'],required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
    cfg,cache=validate_stage(a.stage,'MoGERNN'); set_seed(a.seed); device=require_gpu(); c=cfg['runners']['MoGERNN']
    if int(c['batch_size']) != 1:
        raise RuntimeError('MoGERNN batch_size must remain 1 for sample-varying masks.')
    tr=MoGERNNDataset(cache,'train'); va=MoGERNNDataset(cache,'val'); N=int(tr.x.shape[2])
    adj=torch.from_numpy(np.asarray(np.load(cache/'adjacency.npy'),np.float32)).to(device)
    if tuple(adj.shape)!=(N,N): raise RuntimeError(f'adjacency shape {tuple(adj.shape)} != {(N,N)}')
    model=build_model(cfg,device); nparams=sum(p.numel() for p in model.parameters() if p.requires_grad)
    trl=DataLoader(tr,batch_size=1,shuffle=True,num_workers=0,pin_memory=True); val=DataLoader(va,batch_size=1,shuffle=False,num_workers=0,pin_memory=True)
    opt=torch.optim.AdamW(model.parameters(),lr=float(c['learning_rate']),weight_decay=float(c['weight_decay']))
    scheduler=None
    if str(c.get('scheduler','plateau')).lower() == 'plateau':
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,mode='min',factor=float(c.get('scheduler_factor',0.5)),
            patience=int(c.get('scheduler_patience',1)),
            min_lr=float(c.get('min_learning_rate',1e-5)),
        )
    eps=float(c.get('epsilon_for_observed_zero',1e-6))
    atol=float(c.get('eval_target_independence_atol',1e-6))
    contract_result=None
    if bool(c.get('require_eval_target_independence',True)):
        cx,cobs,cy,_cv=next(iter(val))
        cx=_prepare_x(cx.float().to(device),cobs.to(device),eps)
        cy=cy.float().to(device).unsqueeze(-1)
        contract_result=assert_eval_target_independence(model,adj,cx,cy,0,device,atol)
    if a.smoke:
        x,obs,y,v=next(iter(trl)); model.train(); opt.zero_grad(set_to_none=True)
        p=forward_official(model,adj,_prepare_x(x.float().to(device),obs.to(device),eps),y.float().to(device).unsqueeze(-1),0,device).squeeze(-1)
        loss=masked_mse(p,y.float().to(device),v.to(device)); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),float(c['gradient_clip'])); opt.step()
        if tuple(p.shape)!=(1,int(cfg['experiment']['horizon']),N) or not torch.isfinite(p).all(): raise RuntimeError('MoGERNN smoke output contract failed')
        print('MOGERNN GPU SMOKE PASS',{'loss':float(loss),'shape':tuple(p.shape),'params':nparams},flush=True); return
    best=float('inf'); state=None; be=0; stale=0; t0=time.time()
    for ep in range(1,int(c['max_epochs'])+1):
        model.train(); tt=tn=0.0
        for x,obs,y,v in trl:
            x=_prepare_x(x.float().to(device),obs.to(device),eps); y=y.float().to(device); v=v.to(device)
            opt.zero_grad(set_to_none=True); p=forward_official(model,adj,x,y.unsqueeze(-1),ep-1,device).squeeze(-1)
            loss=masked_mse(p,y,v); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),float(c['gradient_clip'])); opt.step(); tt+=float(loss.item()); tn+=1
        model.eval(); vt=vn=0.0
        with torch.no_grad():
            for x,obs,y,v in val:
                x=_prepare_x(x.float().to(device),obs.to(device),eps); y=y.float().to(device); v=v.to(device)
                p=forward_official(model,adj,x,y.unsqueeze(-1),ep-1,device).squeeze(-1); loss=masked_mse(p,y,v); vt+=float(loss.item()); vn+=1
        score=vt/max(vn,1)
        if scheduler is not None: scheduler.step(score)
        lr=float(opt.param_groups[0]['lr'])
        print(f'EPOCH {ep:03d} train_MSE_norm={tt/max(tn,1):.6f} val_MSE_norm={score:.6f} lr={lr:.8g}',flush=True)
        if score < best-1e-8: best=score; be=ep; stale=0; state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
        else:
            stale+=1
            if stale>=int(c['patience']): print(f'EARLY STOP epoch={ep} best_epoch={be}',flush=True); break
    if state is None: raise RuntimeError('MoGERNN produced no checkpoint')
    model.load_state_dict(state); model.to(device)
    if bool(c.get('require_eval_target_independence',True)):
        cx,cobs,cy,_cv=next(iter(val))
        cx=_prepare_x(cx.float().to(device),cobs.to(device),eps)
        cy=cy.float().to(device).unsqueeze(-1)
        contract_result=assert_eval_target_independence(model,adj,cx,cy,max(be-1,0),device,atol)
    mean,std=scale_stats(cache); raw=predict(model,val,adj,device,eps,mean,std,be)
    rd=make_run_dir(cfg,'MoGERNN',a.stage,a.seed); torch.save({
        'model':state,'best_epoch':be,'best_val_mse_norm':best,'config':vars(model_config(cfg)),
        'eval_target_independence':contract_result,
    },rd/'best.pt')
    sm=save_val_outputs(cfg,cache,rd,raw,'MoGERNN',a.seed,a.stage,{
        'best_epoch':be,'best_val_MSE_norm':best,'params':nparams,'wall_seconds':time.time()-t0,
        'official_architecture':'vendor/MoGERNN/model/MoGERNN.py::Model',
        'adaptation':'dynamic TraffiDent mask, batch_size=1, zero sentinel with observed-zero epsilon protection',
        'table_group':'partial_sensing_subtrack','checkpoint_selection':'Val masked MSE only',
        'training_schedule':'AdamW + ReduceLROnPlateau; batch_size=1 only for sample-varying benchmark masks',
        'eval_target_independence':contract_result,'test_access':False,
    })
    print('MOGERNN COMPLETE',json.dumps({'run_dir':str(rd),'val':sm['val']},default=str))


if __name__=='__main__': main()
