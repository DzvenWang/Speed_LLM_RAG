from __future__ import annotations
import copy, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader
from .data import SupportSplit, RetrievalCache
from .stla import load_pretrained_stla, MaskAwareSTLAformer
from .model import MaskRAFTSTLA
from .evaluation import metrics, summarize_by_family, save_summary


def seed_all(seed):
    import random; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

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

def run_epoch(model,loader,rcache,memory_y,device,mean,std,cfg,opt=None,epoch=1):
    train=opt is not None; model.train(train); total=aux=gatev=0.; nrow=0
    ctx=torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for bi,b in enumerate(loader,1):
            idx=b['index'].numpy() if torch.is_tensor(b['index']) else np.asarray(b['index']); ev,conf=rcache.batch_evidence(idx,memory_y,cfg['retrieval']['temperature']); b=_batch_to_device(b,device); ev=ev.to(device); conf=conf.to(device)
            if train: opt.zero_grad(set_to_none=True)
            out=model(b['x'].float(),b['mask'].float(),b['tod'].long(),b['dow'].long(),ev.float(),conf.float())
            main=_loss_raw(out['prediction'],b['y_raw'].float(),b['future_valid'],mean,std,cfg['training']['forecast_loss'])
            raux=_loss_raw(out['retrieval_prediction'],b['y_raw'].float(),b['future_valid'],mean,std,'huber'); greg=out['gate'].mean()
            loss=main+float(cfg['training']['retrieval_aux_weight'])*raux+float(cfg['training']['gate_reg_weight'])*greg
            if train:
                loss.backward(); nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],float(cfg['training']['grad_clip'])); opt.step()
            bs=b['x'].shape[0]; total+=float(main.detach())*bs; aux+=float(raux.detach())*bs; gatev+=float(greg.detach())*bs; nrow+=bs
            if train and bi%int(cfg['runtime']['print_every'])==0: print(f'epoch={epoch:03d} batch={bi:05d}/{len(loader):05d} main={total/nrow:.6f} retr={aux/nrow:.6f} gate={gatev/nrow:.4f}',flush=True)
    return {'forecast_loss':total/max(nrow,1),'retrieval_loss':aux/max(nrow,1),'gate_mean':gatev/max(nrow,1)}

def predict(model,loader,rcache,memory_y,device,mean,std,temperature=0.10):
    preds=[]; ys=[]; valids=[]; directs=[]; gates=[]; model.eval()
    with torch.no_grad():
        for b in loader:
            idx=b['index'].numpy() if torch.is_tensor(b['index']) else np.asarray(b['index']); ev,conf=rcache.batch_evidence(idx,memory_y,temperature); b=_batch_to_device(b,device)
            o=model(b['x'].float(),b['mask'].float(),b['tod'].long(),b['dow'].long(),ev.to(device).float(),conf.to(device).float()); preds.append((o['prediction']*std+mean).cpu().numpy()); ys.append(b['y_raw'].cpu().numpy()); valids.append(b['future_valid'].cpu().numpy()); directs.append(b['direct_mask'].cpu().numpy()); gates.append(o['gate'].mean().item())
    return np.concatenate(preds),np.concatenate(ys),np.concatenate(valids),np.concatenate(directs),float(np.mean(gates))

def train(cfg,cache,ckpt,retrieval_train,retrieval_val,run_dir,smoke=False):
    seed_all(int(cfg['experiment']['seed'])); device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); model,base_ck=build_model(cfg,ckpt,device); mean=float(base_ck['mean']); std=float(base_ck['std'])
    limit=32 if smoke else None; tr=SupportSplit(cache,'train',limit); va=SupportSplit(cache,'val',limit); memory_y=np.load(Path(cache)/'train_y_norm.npy',mmap_mode='r'); rt=RetrievalCache(retrieval_train); rv=RetrievalCache(retrieval_val)
    bs=min(int(cfg['training']['batch_size']),16 if not smoke else 4); dl=lambda ds,sh:DataLoader(ds,batch_size=bs,shuffle=sh,num_workers=int(cfg['runtime']['num_workers']),pin_memory=torch.cuda.is_available())
    trl,val=dl(tr,True),dl(va,False); opt=make_optimizer(model,cfg); best=float('inf'); best_state=None; best_ep=0; stale=0; history=[]; maxep=2 if smoke else int(cfg['training']['max_epochs'])
    for ep in range(1,maxep+1):
        phase=_phase(ep,cfg); model.set_train_phase(phase,int(cfg['training']['partial_last_layers'])); t=time.time(); a=run_epoch(model,trl,rt,memory_y,device,mean,std,cfg,opt,ep); v=run_epoch(model,val,rv,memory_y,device,mean,std,cfg,None,ep)
        rec={'epoch':ep,'phase':phase,**{f'train_{k}':x for k,x in a.items()},**{f'val_{k}':x for k,x in v.items()},'seconds':time.time()-t}; history.append(rec); print('EPOCH '+json.dumps(rec),flush=True)
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
