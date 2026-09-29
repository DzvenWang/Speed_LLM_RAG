from __future__ import annotations
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .adapters import build_adapter
from .data import CachedSplit
from .identity import load_cache_identity, require_cache_version_compatibility
from .metrics import METRICS_REVISION
from .predictions import save_prediction_npz
from .runner import _assert_prediction_shape, _loss, evaluate
from .utils import count_parameters, ensure_dir, log, seed_all, write_json

MODELS=('STAEformer','HimNet','BigST','STDN','STLAformer')

def _optimizer(model, mc):
    common=dict(lr=float(mc['learning_rate']),weight_decay=float(mc.get('weight_decay',0.0)))
    name=str(mc.get('optimizer','adam')).lower()
    if name=='adam': return torch.optim.Adam(model.parameters(),eps=float(mc.get('adam_eps',1e-8)),**common)
    if name=='adamw': return torch.optim.AdamW(model.parameters(),eps=float(mc.get('adam_eps',1e-8)),**common)
    raise ValueError(f'Unsupported optimizer {name}')

def _scheduler(opt, mc):
    name=str(mc.get('scheduler','none')).lower()
    if name=='multistep': return torch.optim.lr_scheduler.MultiStepLR(opt,[int(x) for x in mc.get('milestones',[])],gamma=float(mc.get('lr_gamma',0.1))),False
    if name=='step': return torch.optim.lr_scheduler.StepLR(opt,step_size=int(mc.get('step_size',10)),gamma=float(mc.get('lr_gamma',0.9))),False
    if name=='plateau': return torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=float(mc.get('lr_gamma',0.1)),patience=int(mc.get('scheduler_patience',20)),threshold=float(mc.get('scheduler_threshold',1e-8))),True
    if name in ('none','null',''): return None,False
    raise ValueError(f'Unsupported scheduler {name}')

def _batch_sizes(mc):
    out=[]
    for x in [int(mc['batch_size'])]+[int(v) for v in mc.get('fallback_batch_sizes',[])]:
        if x>0 and x not in out: out.append(x)
    return out

def _train_once(cfg, recipes, model_name, cache_dir, run_dir, vendor_root, bench, mean, std, seed, batch_size):
    seed_all(seed); run_dir=ensure_dir(run_dir); device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mc=dict(recipes['models'][model_name]); mc['batch_size']=int(batch_size); runtime=cfg['runtime']
    ident=load_cache_identity(cache_dir); require_cache_version_compatibility(cfg,ident,'formal')
    target=str(cfg['experiment']['target'])
    write_json(run_dir/'RUN_MANIFEST.json',{**ident,'code_project_version':str(cfg['version']),'model':model_name,'target':target,'seed':seed,'status':'STARTED'})
    train=CachedSplit(cache_dir,'train'); val=CachedSplit(cache_dir,'val')
    train_loader=DataLoader(train,batch_size=batch_size,shuffle=True,num_workers=int(runtime.get('num_workers',0)),pin_memory=torch.cuda.is_available())
    val_loader=DataLoader(val,batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=torch.cuda.is_available())
    adapter=build_adapter(model_name,vendor_root,bench.N,device,mean,std,mc,bench.adj,bench.nodes); model=adapter.model
    opt=_optimizer(model,mc); sched,plateau=_scheduler(opt,mc); params=count_parameters(model)
    log(f'FORMAL_START model={model_name} target={target} seed={seed} device={device} batch_size={batch_size} params={params:,}')
    best=float('inf'); best_epoch=-1; wait=0; history=[]; batches_seen=0; total=time.time()
    max_epochs=int(mc['max_epochs']); patience=int(mc['patience']); grad=float(mc.get('grad_clip',0)); print_every=int(runtime.get('print_every_batches',100))
    for epoch in range(1,max_epochs+1):
        model.train(); losses=[]; e0=time.time()
        for bi,b in enumerate(train_loader,1):
            opt.zero_grad(set_to_none=True); pn=adapter.train_forward(b,batches_seen); _assert_prediction_shape(pn,b['y_raw'],f'{model_name}/train')
            pr=adapter.raw(pn); loss=_loss(pr,b['y_raw'].to(device).float(),b['future_valid'].to(device),mc.get('train_loss','mae'))
            if not torch.isfinite(loss): raise RuntimeError(f'{model_name} non-finite train loss epoch={epoch} batch={bi}')
            loss.backward()
            if grad>0: torch.nn.utils.clip_grad_norm_(model.parameters(),grad)
            opt.step(); batches_seen+=1; losses.append(float(loss.detach().cpu()))
            if bi%print_every==0: log(f'model={model_name} epoch={epoch:03d} batch={bi:05d}/{len(train_loader):05d} train_loss={np.mean(losses[-print_every:]):.6f}')
        _,_,vs=evaluate(adapter,val_loader,val.meta,device); v=float(vs['MAE'])
        if sched is not None:
            if plateau: sched.step(v)
            else: sched.step()
        rec={'epoch':epoch,'train_loss':float(np.mean(losses)),'val_MAE':v,'val_masked_MAE':float(vs['directly_masked_MAE']),'lr':float(opt.param_groups[0]['lr']),'seconds':float(time.time()-e0)}; history.append(rec)
        log(f'model={model_name} epoch={epoch:03d} train_loss={rec["train_loss"]:.6f} val_MAE={v:.6f} val_masked_MAE={rec["val_masked_MAE"]:.6f} lr={rec["lr"]:.3e} time={rec["seconds"]:.1f}s')
        if v < best-1e-8:
            best=v; best_epoch=epoch; wait=0
            ck={'project_version':cfg['version'],'run_identity':ident,'stage':'formal','model':model_name,'target':target,'state_dict':model.state_dict(),'model_config':mc,'num_nodes':bench.N,'mean':mean,'std':std,'seed':seed,'batch_size':batch_size,'best_val_MAE':best,'best_epoch':best_epoch,'recipe_source':mc.get('recipe_source'),'adapter_revision':getattr(adapter,'adapter_revision',None),'adapter_metadata':getattr(adapter,'adapter_metadata',{})}
            torch.save(ck,run_dir/'best.pt')
        else:
            wait+=1
            if wait>=patience: log(f'EARLY_STOP model={model_name} epoch={epoch} patience={patience}'); break
    pd.DataFrame(history).to_csv(run_dir/'history.csv',index=False)
    ck=torch.load(run_dir/'best.pt',map_location=device,weights_only=False); model.load_state_dict(ck['state_dict'],strict=True)
    vp,vdf,vs=evaluate(adapter,val_loader,val.meta,device)
    save_prediction_npz(run_dir/'val_predictions.npz',val.meta.view_id.astype(str).tolist(),vp.astype(np.float32),manifest_path=run_dir/'val_predictions.manifest.json',metadata={'producer':'Backbone','model':model_name,'target':target,'seed':seed,'stage':'formal','run_identity':ident})
    vdf.to_csv(run_dir/'val_metrics.csv',index=False)
    summary={'project_version':cfg['version'],'run_identity':ident,'stage':'formal','model':model_name,'target':target,'seed':seed,'parameters':params,'batch_size':batch_size,'best_epoch':best_epoch,'best_val_MAE':best,'train_seconds_total':float(time.time()-total),'recipe_source':mc.get('recipe_source'),'validation':vs,'metrics_revision':METRICS_REVISION,'test_evaluated':False}
    write_json(run_dir/'summary.json',summary); write_json(run_dir/'RUN_MANIFEST.json',{**ident,'code_project_version':str(cfg['version']),'model':model_name,'target':target,'seed':seed,'status':'FINISHED','best_epoch':best_epoch,'best_val_MAE':best})
    log(f'FORMAL_PASS model={model_name} target={target} seed={seed} best_epoch={best_epoch} val_MAE={vs["MAE"]:.6f}')
    return summary

def _smoke_once(cfg, recipes, model_name, cache_dir, run_dir, vendor_root, bench, mean, std, seed, batch_size, train_batches=2, val_batches=1):
    """Bounded real-Formal smoke: forward/backward/step, validation, checkpoint save/load."""
    seed_all(seed); run_dir=ensure_dir(run_dir); device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mc=dict(recipes['models'][model_name]); mc['batch_size']=int(batch_size); runtime=cfg['runtime']
    ident=load_cache_identity(cache_dir); require_cache_version_compatibility(cfg,ident,'formal')
    target=str(cfg['experiment']['target']); train_batches=max(1,int(train_batches)); val_batches=max(1,int(val_batches))
    train=CachedSplit(cache_dir,'train'); val=CachedSplit(cache_dir,'val')
    train_loader=DataLoader(train,batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=torch.cuda.is_available())
    val_loader=DataLoader(val,batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=torch.cuda.is_available())
    adapter=build_adapter(model_name,vendor_root,bench.N,device,mean,std,mc,bench.adj,bench.nodes); model=adapter.model
    opt=_optimizer(model,mc); grad=float(mc.get('grad_clip',0)); params=count_parameters(model); losses=[]; batches_seen=0
    log(f'SMOKE_START model={model_name} stage=formal target={target} seed={seed} device={device} batch_size={batch_size} train_batches={train_batches} val_batches={val_batches} params={params:,}')
    model.train()
    for bi,b in enumerate(train_loader,1):
        opt.zero_grad(set_to_none=True); pn=adapter.train_forward(b,batches_seen); _assert_prediction_shape(pn,b['y_raw'],f'{model_name}/smoke_train')
        pr=adapter.raw(pn); loss=_loss(pr,b['y_raw'].to(device).float(),b['future_valid'].to(device),mc.get('train_loss','mae'))
        if not torch.isfinite(loss): raise RuntimeError(f'{model_name} non-finite smoke train loss batch={bi}')
        loss.backward()
        if grad>0: torch.nn.utils.clip_grad_norm_(model.parameters(),grad)
        opt.step(); batches_seen+=1; losses.append(float(loss.detach().cpu())); log(f'SMOKE_TRAIN model={model_name} batch={bi}/{train_batches} loss={losses[-1]:.6f}')
        if bi>=train_batches: break
    model.eval(); vals=[]
    with torch.no_grad():
        for bi,b in enumerate(val_loader,1):
            pn=adapter.eval_forward(b); _assert_prediction_shape(pn,b['y_raw'],f'{model_name}/smoke_val'); pr=adapter.raw(pn)
            loss=_loss(pr,b['y_raw'].to(device).float(),b['future_valid'].to(device),mc.get('train_loss','mae'))
            if not torch.isfinite(loss): raise RuntimeError(f'{model_name} non-finite smoke val loss batch={bi}')
            vals.append(float(loss.detach().cpu())); log(f'SMOKE_VAL model={model_name} batch={bi}/{val_batches} loss={vals[-1]:.6f}')
            if bi>=val_batches: break
    ckpath=Path(run_dir)/'smoke.pt'
    torch.save({'project_version':cfg['version'],'run_identity':ident,'stage':'formal','smoke':True,'model':model_name,'target':target,'state_dict':model.state_dict(),'model_config':mc,'num_nodes':bench.N,'mean':mean,'std':std,'seed':seed,'batch_size':batch_size},ckpath)
    loaded=torch.load(ckpath,map_location=device,weights_only=False)
    if loaded.get('model')!=model_name or loaded.get('target')!=target or int(loaded.get('seed',-1))!=int(seed): raise RuntimeError('Smoke checkpoint identity mismatch')
    model.load_state_dict(loaded['state_dict'],strict=True)
    summary={'status':'PASS','project_version':cfg['version'],'stage':'formal','smoke':True,'model':model_name,'target':target,'seed':seed,'device':str(device),'batch_size':batch_size,'train_batches':train_batches,'val_batches':val_batches,'train_loss_mean':float(np.mean(losses)),'val_loss_mean':float(np.mean(vals)),'cache_dir':str(cache_dir),'checkpoint':str(ckpath),'parameters':params}
    write_json(Path(run_dir)/'smoke_summary.json',summary)
    log(f'SMOKE_PASS model={model_name} stage=formal target={target} seed={seed} train_loss={summary["train_loss_mean"]:.6f} val_loss={summary["val_loss_mean"]:.6f} checkpoint={ckpath}')
    return summary

def smoke_model(cfg, recipes, model_name, cache_dir, run_dir, vendor_root, bench, mean, std, seed, train_batches=2, val_batches=1):
    if model_name not in MODELS: raise ValueError(f'Generic smoke supports {MODELS}; PatchSTG uses its dedicated smoke')
    mc=recipes['models'][model_name]
    smoke_bs=int(mc.get('smoke_batch_size', min(int(mc['batch_size']), 4)))
    candidates=[]
    for x in [smoke_bs]+_batch_sizes(mc):
        if x>0 and x not in candidates: candidates.append(x)
    last=None
    for i,bs in enumerate(candidates):
        try:
            if i: log(f'SMOKE_OOM_RETRY model={model_name} batch_size={bs}')
            return _smoke_once(cfg,recipes,model_name,cache_dir,run_dir,vendor_root,bench,mean,std,seed,bs,train_batches,val_batches)
        except torch.cuda.OutOfMemoryError as e:
            last=e; log(f'SMOKE_CUDA_OOM model={model_name} batch_size={bs}')
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            for name in ('smoke.pt','smoke_summary.json'):
                q=Path(run_dir)/name
                if q.exists(): q.unlink()
    if last is not None: raise last
    raise RuntimeError('No usable smoke batch size')

def train_model(cfg, recipes, model_name, cache_dir, run_dir, vendor_root, bench, mean, std, seed):
    if model_name not in MODELS: raise ValueError(f'Generic formal runner supports {MODELS}; PatchSTG uses its dedicated runner')
    last=None
    for i,bs in enumerate(_batch_sizes(recipes['models'][model_name])):
        try:
            if i: log(f'OOM_RETRY model={model_name} batch_size={bs}')
            return _train_once(cfg,recipes,model_name,cache_dir,run_dir,vendor_root,bench,mean,std,seed,bs)
        except torch.cuda.OutOfMemoryError as e:
            last=e; log(f'CUDA_OOM model={model_name} batch_size={bs}')
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            for name in ('best.pt','history.csv','val_predictions.npz','val_predictions.manifest.json','val_metrics.csv','summary.json'):
                p=Path(run_dir)/name
                if p.exists(): p.unlink()
    if last is not None: raise last
    raise RuntimeError('No usable batch size')
