#!/usr/bin/env python3
"""Train the mask-aware PatchSTG and state-reconstruction ablation."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from oracle.backbone import load_patchstg_adapter
from oracle.config import load_config
from oracle.contract import assert_observation_experiment_ready,cache_dir
from oracle.data import OracleCachedDataset
from oracle.deep import make_inputs
from oracle.metrics import MetricAccumulator,improvement_pct
from oracle.observation import build_observation_aware,masked_mae_torch
from oracle.utils import deterministic_split,log,seed_all,sha256_file,write_json


def loader(ds,batch_size,workers,shuffle=False,indices=None,seed=10):
    source=Subset(ds,[int(x) for x in indices]) if indices is not None else ds
    g=torch.Generator(); g.manual_seed(int(seed))
    kw=dict(dataset=source,batch_size=int(batch_size),shuffle=bool(shuffle),drop_last=False,
            num_workers=int(workers),pin_memory=torch.cuda.is_available(),generator=g)
    if int(workers)>0: kw.update(persistent_workers=True,prefetch_factor=2)
    return DataLoader(**kw)


def base_forward(model,b,device):
    x,te=make_inputs(b['x_obs'],b['tod_hist'],b['dow_hist'],device)
    pred=model(x,te)
    return pred.squeeze(-1) if pred.ndim==4 and pred.shape[-1]==1 else pred


def aware_forward(model,b,device,return_state=False):
    x,te=make_inputs(b['x_obs'],b['tod_hist'],b['dow_hist'],device)
    observed=b['history_observed'].to(device,non_blocking=True).bool()
    return model(x,te,observed,return_state=return_state)


@torch.no_grad()
def evaluate(model,dl,device,mean,std,horizon,aware,state_metric=False):
    model.eval(); acc=MetricAccumulator(horizon); state_abs=0.0; fill_abs=0.0; state_n=0
    for b in dl:
        if aware:
            out=aware_forward(model,b,device,return_state=state_metric)
            if state_metric:
                pn,state,_=out
                target=b['x_complete'].to(device,non_blocking=True).float()
                mask=b['restorable'].to(device,non_blocking=True).bool()&torch.isfinite(target)
                if torch.any(mask):
                    state_abs+=float(torch.abs(state-target)[mask].sum().item())
                    fill=b['x_obs'].to(device,non_blocking=True).float()
                    fill_abs+=float(torch.abs(fill-target)[mask].sum().item()); state_n+=int(mask.sum().item())
            else: pn=out
        else: pn=base_forward(model,b,device)
        pred=(pn*float(std)+float(mean)).detach().cpu().numpy()
        acc.update(pred,b['y_raw'].numpy(),b['future_valid'].numpy(),b['direct_mask'].numpy(),b['support_mask'].numpy())
    out=acc.summary()
    if state_metric:
        out['missing_history_reconstruction_mae_norm']=state_abs/state_n if state_n else float('nan')
        out['fill_value_reconstruction_mae_norm']=fill_abs/state_n if state_n else float('nan')
        out['reconstruction_gain_vs_fill_pct']=improvement_pct(fill_abs/state_n,state_abs/state_n) if state_n else float('nan')
        out['missing_history_reconstruction_count']=state_n
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',default='development',choices=['development','formal'])
    ap.add_argument('--run-tag',required=True)
    ap.add_argument('--variant',required=True,choices=['mask_aware','state_aux'])
    ap.add_argument('--checkpoint',default=None)
    a=ap.parse_args()

    cfg=load_config(ROOT); oc=cfg['observation_aware']; vc=oc['variants'][a.variant]
    seed=int(cfg['experiment']['seed']); seed_all(seed)
    ident=assert_observation_experiment_ready(cfg,a.stage); cache=cache_dir(cfg,a.stage)
    out=Path(cfg['project']['exp_root'])/cfg['experiment']['subgraph']/a.stage/a.run_tag/a.variant
    out.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    adapter,base_path,base_ck,source=load_patchstg_adapter(cfg,a.stage,device,a.checkpoint)
    mean,std=float(base_ck['mean']),float(base_ck['std'])
    train_ds=OracleCachedDataset(cache,'train'); val_ds=OracleCachedDataset(cache,'val')
    dev_ids,final_ids=deterministic_split(len(val_ds),cfg['experiment']['dev_queries'],cfg['experiment']['final_val_queries'],seed)
    train_dl=loader(train_ds,oc['batch_size'],oc['num_workers'],True,seed=seed)
    dev_dl=loader(val_ds,oc['eval_batch_size'],oc['num_workers'],False,dev_ids,seed)
    final_dl=loader(val_ds,oc['eval_batch_size'],oc['num_workers'],False,final_ids,seed)
    all_dl=loader(val_ds,oc['eval_batch_size'],oc['num_workers'],False,seed=seed)

    model=build_observation_aware(adapter.model,cfg['experiment']['history'],oc['mask_hidden']).to(device)
    train_state=a.variant=='state_aux'
    model.set_trainable(oc['freeze_backbone'],oc['train_regression_head'],train_state)
    params=[p for p in model.parameters() if p.requires_grad]
    if not params: raise RuntimeError('No trainable parameters')

    # Zero initialization is a hard controlled-experiment requirement.
    first=next(iter(dev_dl)); model.eval(); adapter.model.eval()
    with torch.no_grad():
        reference=base_forward(adapter.model,first,device)
        initial=aware_forward(model,first,device)
        identity_diff=float(torch.max(torch.abs(reference-initial)).item())
    log(f'ZERO_INIT_EQUIVALENCE max_abs_diff={identity_diff:.9g}')
    if identity_diff>1e-6: raise RuntimeError('Observation-aware zero-init does not reproduce the reference PatchSTG')

    opt=torch.optim.AdamW(params,lr=float(oc['learning_rate']),weight_decay=float(oc['weight_decay']))
    state_weight=float(vc['state_loss_weight']); best=float('inf'); best_epoch=-1; bad=0; history=[]
    t0=time.time()
    for epoch in range(1,int(oc['max_epochs'])+1):
        model.train(); total=forecast_total=state_total=0.0; batches=0
        for b in train_dl:
            opt.zero_grad(set_to_none=True)
            if train_state:
                pn,state,_=aware_forward(model,b,device,return_state=True)
            else:
                pn=aware_forward(model,b,device); state=None
            pred_raw=pn*std+mean
            y=b['y_raw'].to(device,non_blocking=True).float()
            valid=b['future_valid'].to(device,non_blocking=True).bool()
            forecast_loss=masked_mae_torch(pred_raw,y,valid)
            if train_state:
                target=b['x_complete'].to(device,non_blocking=True).float()
                restorable=b['restorable'].to(device,non_blocking=True).bool()
                state_loss=masked_mae_torch(state,target,restorable)
            else: state_loss=forecast_loss.new_zeros(())
            loss=forecast_loss+state_weight*state_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,float(oc['grad_clip']))
            opt.step()
            total+=float(loss.item()); forecast_total+=float(forecast_loss.item()); state_total+=float(state_loss.item()); batches+=1
        dev=evaluate(model,dev_dl,device,mean,std,cfg['experiment']['horizon'],True,train_state)
        score=float(dev['masked_mae'])
        row={'epoch':epoch,'train_total':total/max(batches,1),'train_forecast_mae':forecast_total/max(batches,1),
             'train_state_mae_norm':state_total/max(batches,1),'dev':dev}
        history.append(row)
        improved=score < best-float(oc['min_delta'])
        if improved:
            best=score; best_epoch=epoch; bad=0
            torch.save({'format':'patchstg_observation_aware_v1','version':cfg['version'],'variant':a.variant,
                'state_dict':model.state_dict(),'history':int(cfg['experiment']['history']),'mask_hidden':int(oc['mask_hidden']),
                'base_checkpoint':str(base_path),'base_checkpoint_sha256':sha256_file(base_path),
                'base_checkpoint_source':source,'mean':mean,'std':std,'best_epoch':epoch,
                'best_dev_masked_mae':score,'cache_identity':ident,'config':oc},out/'best.pt')
        else: bad+=1
        log(f'EPOCH {epoch:03d} train_forecast={row["train_forecast_mae"]:.6f} train_state={row["train_state_mae_norm"]:.6f} dev_masked={score:.6f} best={best:.6f} bad={bad}')
        if bad>=int(oc['patience']): break

    saved=torch.load(out/'best.pt',map_location=device,weights_only=False); model.load_state_dict(saved['state_dict']); model.eval()
    log(f'MODEL_SELECTION frozen epoch={best_epoch} dev_masked_mae={best:.6f}; opening Final once')
    parts={'all':all_dl,'dev':dev_dl,'final':final_dl}
    baseline={}; adapted={}
    for name,dl in parts.items():
        baseline[name]=evaluate(adapter.model,dl,device,mean,std,cfg['experiment']['horizon'],False)
        adapted[name]=evaluate(model,dl,device,mean,std,cfg['experiment']['horizon'],True,train_state)
    gains={p:{k:improvement_pct(baseline[p][k],adapted[p][k]) for k in
              ['overall_mae','masked_mae','support_mae','overall_rmse','masked_rmse','support_rmse']} for p in parts}
    summary={'version':cfg['version'],'stage':a.stage,'run_tag':a.run_tag,'variant':a.variant,
        'definition':('observation-mask adapter' if a.variant=='mask_aware' else 'observation-mask adapter plus missing-state reconstruction auxiliary loss'),
        'base_checkpoint':str(base_path),'base_checkpoint_sha256':sha256_file(base_path),'base_checkpoint_source':source,
        'cache_identity':ident,'zero_init_max_abs_diff':identity_diff,'trainable_parameters':sum(p.numel() for p in params),
        'selection':{'criterion':'Development masked MAE only','best_epoch':best_epoch,'best_dev_masked_mae':best},
        'split':{'train_queries':len(train_ds),'val_queries':len(val_ds),'dev_queries':len(dev_ids),'final_queries':len(final_ids),'seed':seed},
        'training_history':history,'baseline':baseline,'adapted':adapted,'gains_pct':gains,
        'elapsed_seconds':time.time()-t0}
    write_json(out/'summary.json',summary)
    f0=baseline['final']['masked_mae']; f1=adapted['final']['masked_mae']
    log('================================================================================')
    log(f'FINAL {a.variant}: masked_MAE {f0:.6f} -> {f1:.6f} ({gains["final"]["masked_mae"]:+.3f}%)')
    if train_state: log(f'FINAL missing-state reconstruction MAE(norm)={adapted["final"]["missing_history_reconstruction_mae_norm"]:.6f}')
    log(f'SUMMARY={out/"summary.json"}')
    log('================================================================================')


if __name__=='__main__': main()
