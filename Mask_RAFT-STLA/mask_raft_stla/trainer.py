from __future__ import annotations
import copy, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader
from .data import SupportSplit, RetrievalCache
from .stla import load_pretrained_stla, MaskAwareSTLAformer
from .model import MaskRAFTSTLA
from .evaluation import metrics, save_summary


def seed_all(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def _phase(epoch,cfg):
    if epoch<=int(cfg['training']['warmup_epochs']): return 'warmup'
    if epoch<=int(cfg['training']['partial_epochs']): return 'partial'
    return 'full'

def build_model(cfg,ckpt,device):
    base,ck=load_pretrained_stla(Path(cfg['project']['root']),Path(ckpt),device); stla=MaskAwareSTLAformer(base)
    m=MaskRAFTSTLA(stla,cfg['retrieval']['periods'],cfg['experiment']['horizon'],cfg['model']['latent_gate_dim'],cfg['model']['fusion_hidden'],cfg['model']['dropout'],cfg['model']['gate_bias']).to(device)
    return m,ck

def make_optimizer(model,cfg):
    base=list(model.stla.base.parameters()); ids={id(p) for p in base}; new=[p for p in model.parameters() if id(p) not in ids]
    return torch.optim.AdamW([{'params':new,'lr':float(cfg['training']['lr_new'])},{'params':base,'lr':float(cfg['training']['lr_backbone'])}],weight_decay=float(cfg['training']['weight_decay']))

def _loss_raw(pred_norm,y_raw,valid,mean,std,kind='huber'):
    pred=pred_norm*std+mean; m=valid.bool()
    if kind=='huber': return nn.functional.smooth_l1_loss(pred[m],y_raw[m])
    return (pred[m]-y_raw[m]).abs().mean()

def _batch_to_device(batch,device):
    return {k:(v.to(device,non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()}

def _make_loader(ds,batch_size,shuffle,cfg,smoke=False):
    workers=0 if smoke else int(cfg['runtime'].get('num_workers',0))
    pin=bool(cfg['runtime'].get('pin_memory',True)) and torch.cuda.is_available()
    kw=dict(batch_size=int(batch_size),shuffle=shuffle,num_workers=workers,pin_memory=pin)
    if workers>0:
        kw['persistent_workers']=bool(cfg['runtime'].get('persistent_workers',True))
        kw['prefetch_factor']=int(cfg['runtime'].get('prefetch_factor',2))
    return DataLoader(ds,**kw)

def _prepare_memory_y(cache,device,cfg):
    arr=np.load(Path(cache)/'train_y_norm.npy',mmap_mode='r')
    if device.type=='cuda' and bool(cfg['runtime'].get('preload_retrieval_memory_gpu',True)):
        try:
            t=torch.from_numpy(np.array(arr,np.float32,copy=True)).to(device)
            gb=t.numel()*t.element_size()/1024**3
            print(f'RETRIEVAL_MEMORY_GPU rows={t.shape[0]} size_gb={gb:.3f}',flush=True)
            return t
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower(): raise
            print(f'RETRIEVAL_MEMORY_GPU_FALLBACK reason={str(e).splitlines()[0]}',flush=True)
            torch.cuda.empty_cache()
    print('RETRIEVAL_MEMORY_CPU_MMAP',flush=True)
    return arr

def run_epoch(model,loader,rcache,memory_y,device,mean,std,cfg,opt=None,epoch=1):
    train=opt is not None; model.train(train); total=aux=gatev=0.; nrow=0
    ctx=torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for bi,b in enumerate(loader,1):
            idx=b['index'].numpy() if torch.is_tensor(b['index']) else np.asarray(b['index'])
            ev,conf=rcache.batch_evidence(idx,memory_y,cfg['retrieval']['temperature'])
            b=_batch_to_device(b,device)
            if not ev.is_cuda: ev=ev.to(device,non_blocking=True)
            if not conf.is_cuda: conf=conf.to(device,non_blocking=True)
            if train: opt.zero_grad(set_to_none=True)
            out=model(b['x'].float(),b['mask'].float(),b['tod'].long(),b['dow'].long(),ev.float(),conf.float())
            main=_loss_raw(out['prediction'],b['y_raw'].float(),b['future_valid'],mean,std,cfg['training']['forecast_loss'])
            raux=_loss_raw(out['retrieval_prediction'],b['y_raw'].float(),b['future_valid'],mean,std,'huber'); greg=out['gate'].mean()
            loss=main+float(cfg['training']['retrieval_aux_weight'])*raux+float(cfg['training']['gate_reg_weight'])*greg
            if train:
                loss.backward(); nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],float(cfg['training']['grad_clip'])); opt.step()
            bs=b['x'].shape[0]; total+=float(main.detach())*bs; aux+=float(raux.detach())*bs; gatev+=float(greg.detach())*bs; nrow+=bs
            if train and bi%int(cfg['runtime']['print_every'])==0:
                print(f'epoch={epoch:03d} batch={bi:05d}/{len(loader):05d} main={total/nrow:.6f} retr={aux/nrow:.6f} gate={gatev/nrow:.4f}',flush=True)
    return {'forecast_loss':total/max(nrow,1),'retrieval_loss':aux/max(nrow,1),'gate_mean':gatev/max(nrow,1)}

def predict(model,loader,rcache,memory_y,device,mean,std,temperature=0.10):
    preds=[]; ys=[]; valids=[]; directs=[]; gates=[]; model.eval()
    with torch.no_grad():
        for b in loader:
            idx=b['index'].numpy() if torch.is_tensor(b['index']) else np.asarray(b['index']); ev,conf=rcache.batch_evidence(idx,memory_y,temperature); b=_batch_to_device(b,device)
            if not ev.is_cuda: ev=ev.to(device,non_blocking=True)
            if not conf.is_cuda: conf=conf.to(device,non_blocking=True)
            o=model(b['x'].float(),b['mask'].float(),b['tod'].long(),b['dow'].long(),ev.float(),conf.float()); preds.append((o['prediction']*std+mean).cpu().numpy()); ys.append(b['y_raw'].cpu().numpy()); valids.append(b['future_valid'].cpu().numpy()); directs.append(b['direct_mask'].cpu().numpy()); gates.append(o['gate'].mean().item())
    return np.concatenate(preds),np.concatenate(ys),np.concatenate(valids),np.concatenate(directs),float(np.mean(gates))

def train(cfg,cache,ckpt,retrieval_train,retrieval_val,run_dir,smoke=False):
    seed_all(int(cfg['experiment']['seed'])); device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); model,base_ck=build_model(cfg,ckpt,device); mean=float(base_ck['mean']); std=float(base_ck['std'])
    limit=32 if smoke else None; tr=SupportSplit(cache,'train',limit); va=SupportSplit(cache,'val',limit); memory_y=_prepare_memory_y(cache,device,cfg); rt=RetrievalCache(retrieval_train); rv=RetrievalCache(retrieval_val)
    if torch.is_tensor(memory_y) and memory_y.is_cuda: rt.to(device); rv.to(device)
    # v1.1: no hidden formal cap at 16. YAML/CLI batch size is authoritative.
    bs=4 if smoke else int(cfg['training']['batch_size'])
    trl=_make_loader(tr,bs,True,cfg,smoke); val=_make_loader(va,bs,False,cfg,smoke)
    print(f'RUNTIME batch_size={bs} train_batches={len(trl)} val_batches={len(val)} num_workers={0 if smoke else int(cfg["runtime"].get("num_workers",0))} pin_memory={bool(cfg["runtime"].get("pin_memory",True))}',flush=True)
    opt=make_optimizer(model,cfg); best=float('inf'); best_state=None; best_ep=0; stale=0; history=[]; maxep=2 if smoke else int(cfg['training']['max_epochs'])
    for ep in range(1,maxep+1):
        phase=_phase(ep,cfg); model.set_train_phase(phase,int(cfg['training']['partial_last_layers']))
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
        t=time.time(); a=run_epoch(model,trl,rt,memory_y,device,mean,std,cfg,opt,ep); v=run_epoch(model,val,rv,memory_y,device,mean,std,cfg,None,ep)
        if device.type=='cuda': torch.cuda.synchronize(device)
        elapsed=time.time()-t; throughput=(len(tr)+len(va))/max(elapsed,1e-9)
        rec={'epoch':ep,'phase':phase,**{f'train_{k}':x for k,x in a.items()},**{f'val_{k}':x for k,x in v.items()},'seconds':elapsed,'samples_per_sec':throughput}
        if device.type=='cuda': rec.update({'gpu_peak_allocated_gb':torch.cuda.max_memory_allocated(device)/1024**3,'gpu_peak_reserved_gb':torch.cuda.max_memory_reserved(device)/1024**3})
        history.append(rec); print('EPOCH '+json.dumps(rec),flush=True)
        score=v['forecast_loss']
        if score<best-float(cfg['training']['min_delta']): best=score;best_ep=ep;stale=0;best_state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
        else:
            stale+=1
            if stale>=int(cfg['training']['patience']): break
    if best_state is None: raise RuntimeError('No checkpoint selected')
    run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=True); pd.DataFrame(history).to_csv(run_dir/'history.csv',index=False); model.load_state_dict(best_state); model.to(device)
    vp,vy,vv,vd,vg=predict(model,val,rv,memory_y,device,mean,std,float(cfg['retrieval']['temperature'])); vm=metrics(vp,vy,vv,vd)
    torch.save({'version':cfg['version'],'model':'Mask_RAFT-STLA','target':cfg['experiment']['target'],'seed':cfg['experiment']['seed'],'state_dict':best_state,'backbone_checkpoint':str(ckpt),'backbone_model_config':base_ck['model_config'],'num_nodes':base_ck['num_nodes'],'mean':mean,'std':std,'best_epoch':best_ep,'best_val_loss':best,'config':cfg},run_dir/'best.pt')
    np.savez_compressed(run_dir/'val_predictions.npz',prediction=vp.astype(np.float32)); save_summary(run_dir/'summary.json',{'model':'Mask_RAFT-STLA','best_epoch':best_ep,'validation':vm,'val_gate_mean':vg,'backbone_checkpoint':str(ckpt),'test_evaluated':False})
    print(f'TRAIN_PASS run_dir={run_dir} best_epoch={best_ep} val_MAE={vm["MAE"]:.6f}',flush=True); return run_dir/'best.pt'
