from __future__ import annotations
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .data import CachedSplit
from .patchstg import build_patchstg, save_layout_snapshot
from .metrics import per_query_metrics, backbone_metric_summary, METRICS_REVISION
from .predictions import save_prediction_npz
from .utils import count_parameters, ensure_dir, log, seed_all, write_json
from .identity import load_cache_identity, require_cache_version_compatibility


def _assert_prediction_shape(pred, target, context):
    if tuple(pred.shape) != tuple(target.shape):
        raise RuntimeError(f'{context} prediction/target shape mismatch pred={tuple(pred.shape)} target={tuple(target.shape)}')


def _loss(pred_raw, y_raw, valid, kind='mae'):
    m = valid.bool() & torch.isfinite(pred_raw) & torch.isfinite(y_raw)
    if not torch.any(m): return (pred_raw * 0).sum()
    p = pred_raw[m]; y = y_raw[m]; kind = str(kind).lower()
    if kind in ('mae','l1'): return torch.abs(p-y).mean()
    if kind in ('huber','smooth_l1'): return F.huber_loss(p,y,reduction='mean',delta=1.0)
    raise KeyError(kind)


def make_optimizer(model, mc):
    common = dict(lr=float(mc['learning_rate']), weight_decay=float(mc.get('weight_decay',0.0)))
    name = str(mc.get('optimizer','adamw')).lower()
    if name == 'adamw': return torch.optim.AdamW(model.parameters(), eps=float(mc.get('adam_eps',1e-8)), **common)
    if name == 'adam': return torch.optim.Adam(model.parameters(), eps=float(mc.get('adam_eps',1e-8)), **common)
    raise ValueError(name)


def make_scheduler(optimizer, mc):
    name = str(mc.get('scheduler','none')).lower()
    if name == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(x) for x in mc.get('milestones',[])], gamma=float(mc.get('lr_gamma',0.5)))
    if name in ('none','null',''): return None
    raise ValueError(name)


@torch.no_grad()
def evaluate(adapter, loader, meta, device):
    adapter.model.eval(); preds=[]; ys=[]; valids=[]; directs=[]; supports=[]; times=[]
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(device)
    for batch in loader:
        if torch.cuda.is_available(): torch.cuda.synchronize(device)
        t0=time.perf_counter(); pn=adapter.eval_forward(batch); _assert_prediction_shape(pn,batch['y_raw'],'PatchSTG/eval'); pr=adapter.raw(pn)
        if torch.cuda.is_available(): torch.cuda.synchronize(device)
        times.append(time.perf_counter()-t0); preds.append(pr.detach().cpu().numpy()); ys.append(batch['y_raw'].numpy()); valids.append(batch['future_valid'].numpy()); directs.append(batch['direct_mask'].numpy()); supports.append(batch['support_mask'].numpy())
    pred=np.concatenate(preds); y=np.concatenate(ys); valid=np.concatenate(valids); direct=np.concatenate(directs); support=np.concatenate(supports)
    rows=per_query_metrics(pred,y,valid,direct,support); df=pd.concat([meta.reset_index(drop=True),pd.DataFrame(rows)],axis=1)
    summary=backbone_metric_summary(pred,y,valid,direct,support)
    fam_values=np.asarray(meta['family'].astype(str))
    for fam in pd.unique(fam_values):
        ii=np.flatnonzero(fam_values==str(fam)); fs=backbone_metric_summary(pred[ii],y[ii],valid[ii],direct[ii],support[ii])
        for c,v in fs.items(): summary[f'{c}__{fam}']=float(v)
    summary['inference_seconds_total']=float(sum(times)); summary['inference_ms_per_query']=float(1000*sum(times)/max(len(df),1)); summary['peak_gpu_memory_mb']=float(torch.cuda.max_memory_allocated(device)/1024**2) if torch.cuda.is_available() else 0.0
    return pred,df,summary


def _candidate_batch_sizes(mc):
    vals=[int(mc['batch_size'])]+[int(x) for x in mc.get('fallback_batch_sizes',[])]; out=[]
    for x in vals:
        if x>0 and x not in out: out.append(x)
    return out


def _train_once(cfg, cache_dir, run_dir, vendor_root, benchmark_adj, benchmark_nodes, mean, std, seed, stage, batch_size):
    seed_all(seed); run_dir=ensure_dir(run_dir); device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); mc=dict(cfg['model']); mc['batch_size']=int(batch_size); runtime=cfg['runtime']
    run_identity=load_cache_identity(cache_dir); require_cache_version_compatibility(cfg,run_identity,stage)
    write_json(run_dir/'RUN_MANIFEST.json',{**run_identity,'code_project_version':str(cfg['version']),'model':'PatchSTG','seed':int(seed),'status':'STARTED'})
    train=CachedSplit(cache_dir,'train'); val=CachedSplit(cache_dir,'val')
    train_loader=DataLoader(train,batch_size=batch_size,shuffle=True,num_workers=int(runtime['num_workers']),pin_memory=torch.cuda.is_available())
    val_loader=DataLoader(val,batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=torch.cuda.is_available())
    adapter=build_patchstg(vendor_root,train.x.shape[2],device,mean,std,mc,benchmark_adj,benchmark_nodes); model=adapter.model
    save_layout_snapshot(run_dir/'spatial_layout.npz', adapter.spatial_layout)
    optimizer=make_optimizer(model,mc); scheduler=make_scheduler(optimizer,mc); params=count_parameters(model)
    lock_path=Path(vendor_root).parent/'SOURCE_LOCK.json'; vendor=json.loads(lock_path.read_text())['repositories']['PatchSTG']
    log(f'stage={stage} model=PatchSTG seed={seed} device={device} batch_size={batch_size} params={params:,} vendor_commit={vendor["commit"]} adapter_revision={adapter.adapter_revision}')
    best=float('inf'); best_epoch=-1; wait=0; history=[]; batches_seen=0; total_start=time.time(); max_epochs=int(mc['max_epochs']); patience=int(mc['patience']); grad=float(mc.get('grad_clip',0.0)); print_every=int(runtime['print_every_batches'])
    for epoch in range(1,max_epochs+1):
        model.train(); losses=[]; e0=time.time()
        for bi,b in enumerate(train_loader,1):
            optimizer.zero_grad(set_to_none=True); pn=adapter.train_forward(b,batches_seen); _assert_prediction_shape(pn,b['y_raw'],'PatchSTG/train'); pr=adapter.raw(pn); y=b['y_raw'].to(device).float(); valid=b['future_valid'].to(device); loss=_loss(pr,y,valid,mc.get('train_loss','mae')); loss.backward()
            if grad>0: torch.nn.utils.clip_grad_norm_(model.parameters(),grad)
            optimizer.step(); losses.append(float(loss.detach().cpu())); batches_seen+=1
            if bi%print_every==0: log(f'epoch={epoch:03d} batch={bi:05d}/{len(train_loader):05d} train_loss={np.mean(losses[-print_every:]):.6f}')
        if scheduler is not None and epoch % max(1,int(mc.get('scheduler_step_interval',1))) == 0: scheduler.step()
        _,_,vs=evaluate(adapter,val_loader,val.meta,device); v=float(vs['MAE'])
        rec={'epoch':epoch,'train_loss':float(np.mean(losses)),'val_MAE':v,'val_support_dependent_MAE':float(vs['support_dependent_MAE']),'lr':float(optimizer.param_groups[0]['lr']),'seconds':float(time.time()-e0)}; history.append(rec)
        log(f'epoch={epoch:03d} train_loss={rec["train_loss"]:.6f} val_MAE={v:.6f} val_support_MAE={rec["val_support_dependent_MAE"]:.6f} lr={rec["lr"]:.3e} time={rec["seconds"]:.1f}s')
        if v < best-1e-8:
            best=v; best_epoch=epoch; wait=0
            torch.save({'project_version':cfg['version'],'run_identity':run_identity,'stage':stage,'model':'PatchSTG','target':str(cfg['experiment'].get('target')),'state_dict':model.state_dict(),'model_config':mc,'num_nodes':train.x.shape[2],'mean':mean,'std':std,'seed':seed,'batch_size':batch_size,'vendor':vendor,'adapter_revision':adapter.adapter_revision,'adapter_metadata':adapter.adapter_metadata,'spatial_layout':adapter.spatial_layout,'best_val_MAE':best,'best_epoch':best_epoch},run_dir/'best.pt')
        else:
            wait+=1
            if wait>=patience: log(f'early_stop epoch={epoch} patience={patience}'); break
    pd.DataFrame(history).to_csv(run_dir/'history.csv',index=False)
    ck=torch.load(run_dir/'best.pt',map_location=device,weights_only=False); model.load_state_dict(ck['state_dict']); vp,vdf,vs=evaluate(adapter,val_loader,val.meta,device)
    save_prediction_npz(run_dir/'val_predictions.npz',val.meta.view_id.astype(str).tolist(),vp.astype(np.float32),manifest_path=run_dir/'val_predictions.manifest.json',metadata={'producer':'Backbone','model':'PatchSTG','seed':int(seed),'stage':stage,'split':'val','code_project_version':str(cfg['version']),'run_identity':run_identity})
    vdf.to_csv(run_dir/'val_metrics.csv',index=False)
    summary={'project_version':cfg['version'],'run_identity':run_identity,'stage':stage,'model':'PatchSTG','target':str(cfg['experiment'].get('target')),'seed':seed,'parameters':params,'batch_size':batch_size,'best_epoch':best_epoch,'best_val_MAE':best,'train_seconds_total':float(time.time()-total_start),'vendor':vendor,'adapter_revision':adapter.adapter_revision,'adapter_metadata':adapter.adapter_metadata,'recipe_source':mc.get('recipe_source'),'validation':vs,'metrics_revision':METRICS_REVISION,'test_evaluated':False}
    write_json(run_dir/'summary.json',summary); write_json(run_dir/'RUN_MANIFEST.json',{**run_identity,'code_project_version':str(cfg['version']),'model':'PatchSTG','seed':int(seed),'status':'FINISHED','best_epoch':int(best_epoch),'best_val_MAE':float(best)})
    log(f'FINISHED stage={stage} PatchSTG seed={seed}: val_MAE={vs["MAE"]:.6f} val_support={vs["support_dependent_MAE"]:.6f}; TEST NOT EVALUATED')
    return summary



def smoke_patchstg(cfg, cache_dir, run_dir, vendor_root, benchmark_adj, benchmark_nodes, mean, std, seed, stage, train_batches=2, val_batches=1):
    """Run a bounded end-to-end training smoke test against the real stage cache."""
    seed_all(seed); run_dir=ensure_dir(run_dir); device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mc=dict(cfg['model']); runtime=cfg['runtime']; batch_size=int(mc.get('smoke_batch_size', min(int(mc['batch_size']), 4)))
    train_batches=max(1,int(train_batches)); val_batches=max(1,int(val_batches))
    run_identity=load_cache_identity(cache_dir); require_cache_version_compatibility(cfg,run_identity,stage)
    train=CachedSplit(cache_dir,'train'); val=CachedSplit(cache_dir,'val')
    train_loader=DataLoader(train,batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=torch.cuda.is_available())
    val_loader=DataLoader(val,batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=torch.cuda.is_available())
    adapter=build_patchstg(vendor_root,train.x.shape[2],device,mean,std,mc,benchmark_adj,benchmark_nodes); model=adapter.model
    optimizer=make_optimizer(model,mc); grad=float(mc.get('grad_clip',0.0)); losses=[]; batches_seen=0
    log(f'SMOKE_START stage={stage} seed={seed} device={device} batch_size={batch_size} train_batches={train_batches} val_batches={val_batches}')
    model.train()
    for bi,b in enumerate(train_loader,1):
        optimizer.zero_grad(set_to_none=True); pn=adapter.train_forward(b,batches_seen); _assert_prediction_shape(pn,b['y_raw'],'PatchSTG/smoke_train')
        pr=adapter.raw(pn); y=b['y_raw'].to(device).float(); valid=b['future_valid'].to(device); loss=_loss(pr,y,valid,mc.get('train_loss','mae'))
        if not torch.isfinite(loss): raise RuntimeError(f'Smoke train loss is non-finite: {float(loss.detach().cpu())}')
        loss.backward()
        if grad>0: torch.nn.utils.clip_grad_norm_(model.parameters(),grad)
        optimizer.step(); losses.append(float(loss.detach().cpu())); batches_seen+=1
        log(f'SMOKE_TRAIN batch={bi}/{train_batches} loss={losses[-1]:.6f}')
        if bi>=train_batches: break
    model.eval(); val_losses=[]
    with torch.no_grad():
        for bi,b in enumerate(val_loader,1):
            pn=adapter.eval_forward(b); _assert_prediction_shape(pn,b['y_raw'],'PatchSTG/smoke_val'); pr=adapter.raw(pn)
            y=b['y_raw'].to(device).float(); valid=b['future_valid'].to(device); loss=_loss(pr,y,valid,mc.get('train_loss','mae'))
            if not torch.isfinite(loss): raise RuntimeError(f'Smoke val loss is non-finite: {float(loss.detach().cpu())}')
            val_losses.append(float(loss.detach().cpu())); log(f'SMOKE_VAL batch={bi}/{val_batches} loss={val_losses[-1]:.6f}')
            if bi>=val_batches: break
    smoke_ck=run_dir/'smoke.pt'; run_dir.mkdir(parents=True,exist_ok=True)
    torch.save({'project_version':cfg['version'],'run_identity':run_identity,'stage':stage,'smoke':True,'model':'PatchSTG','state_dict':model.state_dict(),'target':cfg['experiment'].get('target'),'seed':int(seed),'batch_size':batch_size},smoke_ck)
    loaded=torch.load(smoke_ck,map_location=device,weights_only=False)
    if loaded.get('target') != cfg['experiment'].get('target'): raise RuntimeError('Smoke checkpoint target identity mismatch')
    write_json(run_dir/'smoke_summary.json',{'status':'PASS','stage':stage,'target':cfg['experiment'].get('target'),'seed':int(seed),'device':str(device),'batch_size':batch_size,'train_batches':train_batches,'val_batches':val_batches,'train_loss_mean':float(np.mean(losses)),'val_loss_mean':float(np.mean(val_losses)),'cache_dir':str(cache_dir),'checkpoint':str(smoke_ck)})
    log(f'SMOKE_PASS stage={stage} target={cfg["experiment"].get("target")} seed={seed} train_loss={np.mean(losses):.6f} val_loss={np.mean(val_losses):.6f} checkpoint={smoke_ck}')
    return {'status':'PASS','checkpoint':str(smoke_ck)}

def train_patchstg(cfg, cache_dir, run_dir, vendor_root, benchmark_adj, benchmark_nodes, mean, std, seed, stage):
    last=None
    for idx,bs in enumerate(_candidate_batch_sizes(cfg['model'])):
        try:
            if idx: log(f'Retrying PatchSTG with memory fallback batch_size={bs}')
            return _train_once(cfg,cache_dir,run_dir,vendor_root,benchmark_adj,benchmark_nodes,mean,std,seed,stage,bs)
        except torch.cuda.OutOfMemoryError as e:
            last=e; log(f'CUDA OOM at batch_size={bs}')
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            for name in ['best.pt','history.csv','val_predictions.npz','val_predictions.manifest.json','val_metrics.csv','summary.json']:
                p=Path(run_dir)/name
                if p.exists(): p.unlink()
    if last is not None: raise last
    raise RuntimeError('No usable batch size')
