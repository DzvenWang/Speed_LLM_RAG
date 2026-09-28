#!/usr/bin/env python3
from __future__ import annotations
import argparse,copy,json,sys,time
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
ROOT=next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd());sys.path.insert(0,str(ROOT))
from baseline.runners.common import ImputationDataset,make_run_dir,require_gpu,save_val_outputs,set_seed,validate_stage
from baseline.backbones.patchstg_bridge import FrozenPatchSTG
from baseline.compat.imputeformer_tsl_shim import install_imputeformer_tsl_shim

def import_model():
    exp=ROOT/'vendor/ImputeFormer/experiments'
    bundled_tsl=exp/'tsl/__init__.py'
    bundled_model=exp/'imputeformer/models/__init__.py'
    if not bundled_tsl.exists():
        raise RuntimeError(f'Missing bundled ImputeFormer TSL implementation: {bundled_tsl}')
    if not bundled_model.exists():
        raise RuntimeError(f'Missing bundled ImputeFormer model package: {bundled_model}')
    if str(exp) not in sys.path:
        sys.path.insert(0,str(exp))
    shim = install_imputeformer_tsl_shim()
    try:
        from imputeformer.models import ImputeFormerModel
    except Exception as e:
        raise RuntimeError(
            'ImputeFormer import failed after installing the minimal TSL '
            'compatibility shim. '+repr(e)
        ) from e
    print('IMPUTEFORMER TSL COMPAT', shim, flush=True)
    return ImputeFormerModel

def build_model(cfg,N):
    c=cfg['runners']['ImputeFormer'];M=import_model()
    return M(num_nodes=N,input_dim=3,output_dim=1,input_embedding_dim=int(c['input_embedding_dim']),learnable_embedding_dim=int(c['learnable_embedding_dim']),feed_forward_dim=int(c['feed_forward_dim']),num_temporal_heads=int(c['num_temporal_heads']),num_layers=int(c['num_layers']),dropout=float(c['dropout']),windows=int(cfg['experiment']['history']),dim_proj=int(c['dim_proj']))

def batch_tensors(b,device):
    idx,x,full,obs,art,tod,dow=b
    x=x.float().to(device);full=full.float().to(device);obs=obs.float().to(device);art=art.bool().to(device);tod=tod.float().to(device);dow=dow.float().to(device)
    u=torch.stack([tod/287.0,dow/6.0],dim=-1).unsqueeze(2).expand(x.shape[0],x.shape[1],x.shape[2],2)
    return idx,x,full,obs,art,tod.long(),dow.long(),u

def reconstruction_loss(pred,full,art):
    p=pred.squeeze(-1);m=art.bool()
    if not torch.any(m): return (p-full).abs().mean()*0
    return (p[m]-full[m]).abs().mean()

def completed_history(model,b,device):
    idx,x,full,obs,art,tod,dow,u=batch_tensors(b,device)
    pred=model(x.unsqueeze(-1),u,obs.unsqueeze(-1)).squeeze(-1)
    comp=torch.where(obs.bool(),x,pred)
    return idx,comp,tod,dow

def forecast_val(model,dl,patch,device):
    out=[];model.eval()
    with torch.no_grad():
        for b in dl:
            _idx,comp,tod,dow=completed_history(model,b,device);out.append(patch.predict_raw(comp,tod,dow).cpu().numpy())
    return np.concatenate(out,0).astype(np.float32)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['development','formal'],default='development');ap.add_argument('--seed',type=int,required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    cfg,cache=validate_stage(a.stage,'ImputeFormer');set_seed(a.seed);device=require_gpu();patch=FrozenPatchSTG(cfg,device);c=cfg['runners']['ImputeFormer']
    tr=ImputationDataset(cache,'train');va=ImputationDataset(cache,'val');N=tr.x.shape[2];model=build_model(cfg,N).to(device);nparams=sum(p.numel() for p in model.parameters() if p.requires_grad)
    dl=lambda ds,sh:DataLoader(ds,batch_size=int(c['batch_size']),shuffle=sh,num_workers=0,pin_memory=True)
    trl,val=dl(tr,True),dl(va,False);opt=torch.optim.Adam(model.parameters(),lr=float(c['learning_rate']),weight_decay=float(c['weight_decay']))
    if a.smoke:
        # 1) verify the official ImputeFormer path can train for one step;
        # 2) verify the completed history can actually be consumed by the
        #    frozen PatchSTG reference on GPU.  The second check is important:
        #    a standalone imputer smoke is not sufficient for this controlled
        #    reconstruction-then-forecast baseline.
        b=next(iter(trl))
        opt.zero_grad(set_to_none=True)
        _,x,full,obs,art,_tod,_dow,u=batch_tensors(b,device)
        p=model(x.unsqueeze(-1),u,obs.unsqueeze(-1))
        loss=reconstruction_loss(p,full,art)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),float(c['gradient_clip']))
        opt.step()

        # Keep the bridge smoke deliberately tiny while exercising the real
        # FrozenPatchSTG adapter, time features and output contract.
        bs=min(2,int(x.shape[0]))
        b_small=tuple(v[:bs] for v in b)
        model.eval()
        with torch.no_grad():
            _idx,comp,tod,dow=completed_history(model,b_small,device)
            pred_raw=patch.predict_raw(comp,tod,dow)
        expected=(bs,int(cfg['experiment']['horizon']),N)
        if tuple(pred_raw.shape)!=expected:
            raise RuntimeError(f'ImputeFormer -> Frozen PatchSTG smoke shape {tuple(pred_raw.shape)} != {expected}')
        if not torch.isfinite(pred_raw).all():
            raise RuntimeError('ImputeFormer -> Frozen PatchSTG smoke produced non-finite predictions')
        print('IMPUTEFORMER GPU SMOKE PASS',float(loss),flush=True)
        print('IMPUTEFORMER -> FROZEN PATCHSTG GPU SMOKE PASS',
              {'prediction_shape':tuple(pred_raw.shape),
               'patchstg_reference_seed':patch.reference_seed,
               'patchstg_checkpoint':str(patch.checkpoint)},flush=True)
        return
    best=float('inf');state=None;be=0;stale=0;t0=time.time()
    for ep in range(1,int(c['max_epochs'])+1):
        model.train();tot=den=0
        for b in trl:
            opt.zero_grad(set_to_none=True);_,x,full,obs,art,_tod,_dow,u=batch_tensors(b,device);p=model(x.unsqueeze(-1),u,obs.unsqueeze(-1));loss=reconstruction_loss(p,full,art);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),float(c['gradient_clip']));opt.step();tot+=float(loss.item());den+=1
        model.eval();vt=vn=0
        with torch.no_grad():
            for b in val:
                _,x,full,obs,art,_tod,_dow,u=batch_tensors(b,device);loss=reconstruction_loss(model(x.unsqueeze(-1),u,obs.unsqueeze(-1)),full,art);vt+=float(loss.item());vn+=1
        score=vt/max(vn,1);print(f'EPOCH {ep:03d} train_impute_MAE_norm={tot/max(den,1):.6f} val_impute_MAE_norm={score:.6f}',flush=True)
        if score<best-1e-8:best=score;be=ep;stale=0;state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
        else:
            stale+=1
            if stale>=int(c['patience']):print(f'EARLY STOP epoch={ep} best_epoch={be}');break
    model.load_state_dict(state);model.to(device);raw=forecast_val(model,val,patch,device);rd=make_run_dir(cfg,'ImputeFormer',a.stage,a.seed)
    torch.save({'model':state,'best_epoch':be,'best_val_impute_mae_norm':best,'patchstg_reference_seed':patch.reference_seed,'patchstg_checkpoint':str(patch.checkpoint)},rd/'best.pt')
    sm=save_val_outputs(cfg,cache,rd,raw,'ImputeFormer',a.seed,a.stage,{'best_epoch':be,'best_val_impute_MAE_norm':best,'params':nparams,'wall_seconds':time.time()-t0,'official_architecture':'vendor/ImputeFormer/experiments/imputeformer/models/ImputeFormer.py::ImputeFormerModel','downstream_forecaster':'Frozen PatchSTG v1.0.10 reference checkpoint','patchstg_reference_seed':patch.reference_seed,'patchstg_checkpoint':str(patch.checkpoint),'checkpoint_selection':'history-imputation validation only','test_access':False})
    print('IMPUTEFORMER COMPLETE',json.dumps({'run_dir':str(rd),'val':sm['val']},default=str))
if __name__=='__main__':main()
