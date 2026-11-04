from __future__ import annotations
import csv,json,os,time,random,signal,copy,math,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .config import json_write,sha256


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def to_device(batch,device):
    return {k:(v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}


def safe_div(a,b):return a/max(float(b),1e-9)


def compute_loss(batch, output, cfg, target_std=1.):
    pred=output['prediction'];truth=batch['y_norm']
    valid=batch['future_valid'].bool()&torch.isfinite(truth)
    if pred.shape!=truth.shape:raise RuntimeError(f'Prediction/label shape mismatch {pred.shape} vs {truth.shape}')
    if not torch.isfinite(pred).all():raise FloatingPointError('Nonfinite predictions')
    count=valid.sum().clamp_min(1)
    err=torch.where(valid,pred-truth,torch.zeros_like(pred))
    mae=err.abs().sum()/count
    rmse=(err.square().sum()/count).clamp_min(0).sqrt()
    direct=batch['direct_mask'].bool()
    if direct.ndim==2:direct=direct[:,None,:].expand_as(valid)
    elif direct.shape!=valid.shape:raise ValueError('Unexpected direct_mask shape')
    dm=valid&direct
    masked=err.abs().masked_fill(~dm,0).sum()/dm.sum().clamp_min(1)
    dv=valid[:,1:]&valid[:,:-1]
    de=(pred[:,1:]-pred[:,:-1])-(truth[:,1:]-truth[:,:-1])
    evolution=de.abs().masked_fill(~dv,0).sum()/dv.sum().clamp_min(1)
    lc=cfg['training']['loss']
    return (mae+float(lc['rmse'])*rmse+float(lc['masked'])*masked+float(lc['evolution'])*evolution),{'mae':float(mae.detach()),'rmse':float(rmse.detach()),'masked':float(masked.detach()),'evolution':float(evolution.detach())}


class MetricAccumulator:
    def __init__(self):self.sums=dict(abs=0.,sq=0.,count=0,masked_abs=0.,masked_count=0,tail_abs=0.,tail_count=0,evo_abs=0.,evo_count=0)
    def update(self,pred,batch,mean,std):
        y=batch['y_raw'];pred=pred.float()*std+mean
        valid=batch['future_valid'].bool()&torch.isfinite(y)
        err=torch.where(valid,pred-y,torch.zeros_like(pred))
        dm=batch['direct_mask'].bool()
        if dm.ndim==2:dm=dm[:,None].expand_as(valid)
        mv=valid&dm
        a=self.sums
        a['abs']+=float(err.abs().sum());a['sq']+=float(err.square().sum());a['count']+=int(valid.sum())
        a['masked_abs']+=float(err.abs().masked_fill(~mv,0).sum());a['masked_count']+=int(mv.sum())
        a['tail_abs']+=float(err[:,-1].abs().sum());a['tail_count']+=int(valid[:,-1].sum())
        de=(pred[:,1:]-pred[:,:-1])-(y[:,1:]-y[:,:-1]);dv=valid[:,1:]&valid[:,:-1]
        a['evo_abs']+=float(de.abs().masked_fill(~dv,0).sum());a['evo_count']+=int(dv.sum())
    def finish(self):
        a=self.sums
        if a['count']==0:raise RuntimeError('No valid targets in evaluation set')
        return {'mae':safe_div(a['abs'],a['count']),'rmse':math.sqrt(safe_div(a['sq'],a['count'])),
                'masked_mae':safe_div(a['masked_abs'],a['masked_count']) if a['masked_count'] else None,
                'h12_mae':safe_div(a['tail_abs'],a['tail_count']),
                'evolution_mae':safe_div(a['evo_abs'],a['evo_count']) if a['evo_count'] else None,
                'valid_count':a['count'],'masked_count':a['masked_count']}


def make_loader(ds,batch_size,workers,shuffle,device):
    kw=dict(batch_size=int(batch_size),shuffle=bool(shuffle),num_workers=int(workers),
            pin_memory=(device.type=='cuda'),drop_last=False)
    if workers:kw.update(persistent_workers=True,prefetch_factor=2)
    return DataLoader(ds,**kw)


def make_trainable(model,stage,cfg):
    model.set_stage(stage)
    # Avoid accumulating gradients through intentionally disabled submodules.
    for p in model.parameters():p.requires_grad_(False)
    modules=[model.encoder.side,model.encoder.mask_proj,model.horizon_prior,model.dynamics.local,model.dynamics.norm,model.head]
    if stage!='direct':modules.extend([model.ocpr,model.evidence_reach])
    if stage in ('transport','coupled'):modules.extend([model.dynamics.trans,model.dynamics.functional])
    for module in modules:
        for p in module.parameters():p.requires_grad_(True)
    if stage in ('transport','coupled'):
        model.dynamics.functional_scale.requires_grad_(True)
    # Pretrained STLA adaptation: train last layers, optionally all in S3.
    if stage=='coupled' and cfg['training'].get('full_encoder_on_coupled',False):
        for p in model.encoder.base.parameters():p.requires_grad_(True)
    else:
        count=int(cfg['training'].get('unfreeze_last_stla_layers',2))
        for layer in list(model.encoder.base.attn_layers_st)[-count:]:
            for p in layer.parameters():p.requires_grad_(True)
    if not any(p.requires_grad for p in model.parameters()):raise RuntimeError('No trainable parameters')


def optimizer_for(model,cfg):
    stla=[];fresh=[]
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        (stla if name.startswith('encoder.base.') else fresh).append(p)
    groups=[]
    if stla:groups.append({'params':stla,'lr':float(cfg['training']['lr_encoder'])})
    if fresh:groups.append({'params':fresh,'lr':float(cfg['training']['lr_new'])})
    return torch.optim.AdamW(groups,weight_decay=float(cfg['training']['weight_decay']))


def scaler_for(device,cfg):
    # Torch 2.x style; older torch is supported by the fallback.
    enabled=device.type=='cuda' and bool(cfg['training'].get('amp',True))
    if hasattr(torch,'amp') and hasattr(torch.amp,'GradScaler'):
        return torch.amp.GradScaler('cuda',enabled=enabled,init_scale=float(cfg['training'].get('initial_amp_scale',1024)))
    return torch.cuda.amp.GradScaler(enabled=enabled,init_scale=float(cfg['training'].get('initial_amp_scale',1024)))


def autocast_for(device,cfg):
    import contextlib
    if device.type=='cuda' and bool(cfg['training'].get('amp',True)):
        return torch.autocast(device_type='cuda',dtype=torch.float16)
    return contextlib.nullcontext()



def crossmask_consistency(model,batch,teacher_state,teacher_reliability,drop_rate=0.15):
    """TRAIN-only submask augmentation; no future labels, no reconstruction from unseen source.

    Compute only on a quarter batch to limit STLA encoder overhead.
    """
    bs=max(1,batch['x_multi'].shape[0]//4)
    x=batch['x_multi'][:bs]
    old_mask=batch['observed_mask'][:bs]
    keep=(torch.rand((bs,1,x.shape[2],1),device=x.device)>=drop_rate).float()
    mask=old_mask*keep
    age=torch.zeros_like(x)
    previous=torch.zeros_like(mask[:,0])
    for t in range(mask.shape[1]):
        previous=(previous+1.)*(1.-mask[:,t]);age[:,t]=previous/mask.shape[1]
    side=torch.where(mask.bool(),x,torch.zeros_like(x))
    x_target=side[...,model.target_idx]
    hist=model.encoder(x_target,side,mask,age,batch['tod'][:bs],batch['dow'][:bs])
    student=model.ocpr(hist,side,mask,age,model.edge_index,model.edge_attr,model.functional_index,model.functional_attr)
    # Distil only where teacher has valid observation support; no hidden-state oracle.
    weights=(old_mask.mean(dim=(1,3))[:bs].unsqueeze(-1)*teacher_reliability[:bs].detach()).clamp(0,1)
    error=(student['state']-teacher_state[:bs].detach()).square().mean(-1,keepdim=True)
    return (weights*error).sum()/weights.sum().clamp_min(1)

def epoch_loop(model,loader,device,cfg,opt=None,scaler=None,limit_batches=None):
    train=opt is not None
    model.train(train)
    if not train:model.eval()
    loss_total=0.;rows=0;scaler_skips=0
    acc_grad=0.
    for i,batch in enumerate(loader):
        if limit_batches and i>=limit_batches:break
        batch=to_device(batch,device)
        if train:opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with autocast_for(device,cfg):
                out=model(batch)
                loss,parts=compute_loss(batch,out,cfg)
        if train:
            scale0=scaler.get_scale()
            scaler.scale(loss).backward()
            cross_weight=float(cfg['training']['loss'].get('crossmask',0.))
            interval=int(cfg['training'].get('crossmask_interval',4))
            if (cross_weight>0 and model.stage in ('observability','coupled')
                    and (i+1)%max(1,interval)==0 and out.get('state') is not None):
                with autocast_for(device,cfg):
                    cross=crossmask_consistency(model,batch,out['state']['state'],out['state']['reliability'])
                scaler.scale(cross_weight*cross).backward()
            scaler.unscale_(opt)
            grad_norm=nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],float(cfg['training']['grad_clip']))
            if not torch.isfinite(grad_norm):
                print('WARN_NONFINITE_GRAD',float(grad_norm),flush=True)
            else:acc_grad+=float(grad_norm)
            scaler.step(opt);scaler.update()
            if scaler.get_scale()<scale0:scaler_skips+=1
        bs=batch['y_norm'].shape[0];loss_total+=float(loss.detach())*bs;rows+=bs
        if train and (i+1)%int(cfg['runtime']['print_every'])==0:
            print(f'TRAIN_STEP stage={model.stage} step={i+1}/{len(loader)} loss={float(loss):.6f} mae_norm={parts["mae"]:.6f} amp_skips={scaler_skips}',flush=True)
    if not rows:raise RuntimeError('No batches completed')
    return {'loss':loss_total/rows,'rows':rows,'amp_skips':scaler_skips,'grad_norm_mean':acc_grad/max(1,len(loader))}


@torch.no_grad()
def evaluate(model,loader,device,cfg,mean,std,intervention=None,max_batches=None):
    model.eval();m=MetricAccumulator();diag=[];support_sum=None;support_n=0
    for i,batch in enumerate(loader):
        if max_batches and i>=max_batches:break
        batch=to_device(batch,device)
        with autocast_for(device,cfg):out=model(batch,intervention=intervention)
        m.update(out['prediction'],batch,mean,std)
        diag.append(out.get('diagnostics',{}))
        if 'future_support' in out:
            sp=out['future_support'].detach().float().mean(dim=2).sum(dim=0)
            support_sum=sp if support_sum is None else support_sum+sp
            support_n+=int(batch['y_norm'].shape[0])
    result=m.finish()
    if support_sum is not None:
        result['mean_support_by_horizon']=(support_sum/max(1,support_n)).cpu().tolist()
    for k in ('transport_norm','lag_entropy'):
        if diag:result[k]=sum(float(d.get(k,0)) for d in diag)/len(diag)
    return result


def rng_state():
    return {'torch':torch.get_rng_state(),'numpy':np.random.get_state(),'python':random.getstate(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(obj):
    torch.set_rng_state(obj['torch']);np.random.set_state(obj['numpy']);random.setstate(obj['python'])
    if obj.get('cuda') is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(obj['cuda'])


def atomic_save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    torch.save(obj,tmp);os.replace(tmp,path)


def build_state(model,opt,scaler,stage,epoch,stage_best,stage_best_epoch,completed,config,identity):
    return {'schema':'pcft_v6_pipeline_v1','stage':stage,'epoch':epoch,
            'model':model.state_dict(),'optimizer':opt.state_dict() if opt else None,
            'scaler':scaler.state_dict() if scaler else None,
            'stage_best':stage_best,'stage_best_epoch':stage_best_epoch,
            'completed':completed,'config':config,'identity':identity,'rng':rng_state()}


def check_identity(state,identity):
    if state.get('schema')!='pcft_v6_pipeline_v1':raise RuntimeError('Wrong v6 checkpoint schema')
    if state.get('identity')!=identity:raise RuntimeError('Checkpoint/data/backbone identity mismatch')


def batch_probe(model,loader,device,cfg,candidates,graph_guard=0.75):
    if device.type!='cuda':return int(candidates[0])
    from torch.utils.data import Subset
    dataset=loader.dataset;choice=int(candidates[0]);scores=[]
    model.set_stage('coupled');make_trainable(model,'coupled',cfg)
    for b in candidates:
        if len(dataset)<b:continue
        dl=make_loader(Subset(dataset,list(range(int(b)))),b,0,False,device)
        try:
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            t=time.perf_counter();tmp=optimizer_for(model,cfg);scaler=scaler_for(device,cfg)
            epoch_loop(model,dl,device,cfg,opt=tmp,scaler=scaler,limit_batches=1)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-t;peak=torch.cuda.max_memory_reserved()/1024**3
            print(f'BATCH_PROBE batch={b} seconds={elapsed:.2f} peak_reserved_gib={peak:.2f}',flush=True)
            if peak<graph_guard*torch.cuda.get_device_properties(device).total_memory/(1024**3):scores.append((b/elapsed,b,peak))
            del tmp,scaler,dl
        except torch.cuda.OutOfMemoryError:
            print(f'BATCH_PROBE_OOM batch={b}',flush=True);torch.cuda.empty_cache()
    if not scores:raise RuntimeError('No batch size passes full-model memory probe')
    choice=max(scores)[1]
    return int(choice)
