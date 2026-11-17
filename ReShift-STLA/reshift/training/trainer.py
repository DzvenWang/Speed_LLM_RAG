"""ReShift staged trainer: normal -> anchor_cf -> [anchor_ret] -> anchor_control.

Shared stages (normal, anchor_cf) are keyed by ReShift-only stage keys and are
reused by every retrieval ablation of the same seed.  Retrieval stages fetch
failure-replayed neighbour evidence from fixed Train mask slots.
Four-severity validation selection, no TEST before lock.
"""
from __future__ import annotations
import gc,json,math,time
import numpy as np
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
from torch.utils.data._utils.collate import default_collate
from reshift.util import (atomic_json,atomic_torch,checkpoint_load,to_device,
                          rng_state,restore_rng,sha256)
from reshift.counterfactual.teacher import PairedTeacher
from reshift.training.losses import compute_loss
from reshift.training.normal_anchor import normal_anchor_loss
from reshift.training.evaluation import evaluate
from reshift.training.layout import retrieval_cfg,RETRIEVAL_STAGES

CF_FIRST_STAGES=('cf','residual','joint','control')
FUSION_ARMS=('ret_res','ret_shift','ret_both','prior')


def stages_for(cfg):
    if cfg['model'].get('prediction_mode')!='normal_anchored':return CF_FIRST_STAGES
    stages=['normal','anchor_cf']
    r=retrieval_cfg(cfg)
    if r and any(a in FUSION_ARMS for a in r.get('arms',())):stages.append('anchor_ret')
    stages.append('anchor_control')
    return tuple(stages)


def load_model_state(model,state,shared):
    """Shared-stage checkpoints may come from a job with a different retrieval
    head; only `retrieval.*` keys may differ and they keep this job's init."""
    if not shared:
        model.load_state_dict(state,strict=True);return
    filt={k:v for k,v in state.items() if not k.startswith('retrieval.')}
    missing,unexpected=model.load_state_dict(filt,strict=False)
    bad=[k for k in missing if not k.startswith('retrieval.')]+list(unexpected)
    if bad:raise RuntimeError(f'Shared checkpoint incompatible beyond retrieval head: {bad[:8]}')


def normal_calibration_partition(ds,config):
    """Time-embargoed *TRAIN* holdout, never complete VAL/TEST Teacher labels."""
    fraction=float(config['training'].get('normal_calibration_fraction',0.12))
    embargo=int(config['training'].get('normal_calibration_embargo',24))
    starts=np.asarray(ds.window_starts,dtype=np.int64)
    if not 0<fraction<.5 or len(starts)<160:return None
    unique=np.unique(starts)
    if len(unique)<160:return None
    threshold=unique[int((1-fraction)*len(unique))]
    train_idx=np.flatnonzero(starts+embargo<threshold)
    cal_idx=np.flatnonzero(starts>=threshold)
    if len(train_idx)<64 or len(cal_idx)<24:return None
    maxcal=int(config['training'].get('normal_calibration_limit',256))
    if maxcal>0 and len(cal_idx)>maxcal:
        cal_idx=cal_idx[np.linspace(0,len(cal_idx)-1,maxcal,dtype=np.int64)]
    if starts[train_idx].max()+embargo>=starts[cal_idx].min():
        raise RuntimeError('Normal calibration temporal embargo violated')
    return train_idx.tolist(),cal_idx.tolist()


@torch.no_grad()
def normal_calibration_score(model,teacher,ds,cal_idx,device,std,eval_batch=32):
    prior=ds.epoch
    ds.set_epoch(0)
    model.eval()
    total=0.;count=0.
    try:
        loader=DataLoader(Subset(ds,cal_idx),batch_size=eval_batch,shuffle=False,num_workers=0)
        for raw in loader:
            b=to_device(raw,device)
            o=model({k:b[k] for k in ('x','mask','tod','dow')},enable_cf=False)
            t=teacher(b,o['h0'],o['backbone'])
            corrected=o['normal_prediction']+t['cf_target']
            valid=b['future_valid'].float()
            y=torch.where(valid.bool(),b['y_norm'].float(),torch.zeros_like(b['y_norm'].float()))
            err=(corrected-y).abs()*std
            total+=float((err*valid).sum());count+=float(valid.sum())
    finally:
        ds.set_epoch(prior)
    return total/max(count,1.)


def uses_evidence(model,stage):
    head=getattr(model,'retrieval',None)
    return head is not None and head.needs_evidence and stage in RETRIEVAL_STAGES


def stage_optimizer(model,stage,cfg):
    tr=cfg['training'];groups=[]
    names={'normal':('failure','residual','graph_residual'),
           'anchor_cf':('cf','input_adapter','graph_cf'),
           'anchor_ret':('retrieval',),
           'anchor_control':('cf','retrieval'),
           'cf':('failure','input_adapter','cf'),
           'residual':('residual',),
           'joint':('failure','input_adapter','cf','residual','graph_cf','graph_residual'),
           'control':('controller',)}[stage]
    for name in names:
        module=getattr(model,name,None)
        if module is None:continue
        params=[p for p in module.parameters() if p.requires_grad]
        if not params:continue
        lr=(tr['lr_cf'] if name in ('cf','failure','input_adapter') else
            tr['lr_residual'] if name=='residual' else
            tr['lr_graph'] if name.startswith('graph') else
            tr.get('lr_retrieval',tr['lr_controller']) if name=='retrieval' else tr['lr_controller'])
        groups.append({'params':params,'lr':float(lr),'name':name})
    used={id(p) for grp in groups for p in grp['params']}
    missing=[n for n,p in model.named_parameters() if p.requires_grad and id(p) not in used]
    if missing:raise RuntimeError('Optimizer missing '+str(missing[:10]))
    if not groups:raise RuntimeError('No trainable parameters')
    return torch.optim.AdamW(groups,weight_decay=float(tr['weight_decay']))


def training_forward(model,b,stage,evidence=None):
    if model.prediction_mode=='normal_anchored':
        return model({k:b[k] for k in ('x','mask','tod','dow')},
                     enable_cf=(stage!='normal'),
                     enable_controller=(stage=='anchor_control'),evidence=evidence)
    return model({k:b[k] for k in ('x','mask','tod','dow')},
                 enable_cf=True,enable_residual=(stage!='cf'),
                 enable_controller=(stage=='control'))


def training_contrast_forward(model,b,stage,cfg):
    """Additional masked views (anchor_cf only); never a future or full input."""
    if not cfg['training'].get('multimask_error_contrast',False):return None
    if model.prediction_mode=='normal_anchored':
        if stage!='anchor_cf':return None
        if 'view_x' not in b or 'view_mask' not in b:
            raise RuntimeError('Normal-anchored K-view training requires view_x/view_mask')
        return [model({'x':b['view_x'][:,k],'mask':b['view_mask'][:,k],'tod':b['tod'],'dow':b['dow']},
                      enable_cf=True,enable_controller=False)
                for k in range(b['view_x'].shape[1])]
    if stage not in ('cf','joint'):return None
    return model({'x':b['contrast_x'],'mask':b['contrast_mask'],'tod':b['tod'],'dow':b['dow']},
                 enable_cf=True,enable_residual=False,enable_controller=False)


def _evidence(provider,model,stage,b,slot):
    if not uses_evidence(model,stage):return None
    if provider is None:raise RuntimeError(f'stage {stage} needs a retrieval EvidenceProvider')
    return provider.batch(provider.train_key(slot),b['index'])


def probe_size(model,teacher,dataset,stage,size,device,cfg,mean,std,optimizer,provider,slot):
    amp=bool(cfg['training']['amp']) and device.type=='cuda'
    if device.type=='cuda':torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
    try:
        b=to_device(default_collate([dataset[0]]*size),device)
        model.train()
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
            ev=_evidence(provider,model,stage,b,slot)
            out=training_forward(model,b,stage,ev)
            alt=training_contrast_forward(model,b,stage,cfg)
            with torch.no_grad():t=teacher(b,out['h0'],out['backbone'])
            loss,_=(normal_anchor_loss(out,b,t,stage,mean,std,cfg,other_views=alt)
                    if model.prediction_mode=='normal_anchored' else
                    compute_loss(out,b,t,stage,mean,std,cfg,contrast=alt))
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss during probe')
        loss.backward()
        if device.type=='cuda':torch.cuda.synchronize(device)
        peak=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0
        capacity=torch.cuda.get_device_properties(device).total_memory if device.type=='cuda' else 0
        margin=max(0.7*1024**3,.12*capacity)
        fits=device.type!='cuda' or (peak+margin<float(cfg['training']['batch_capacity_fraction'])*capacity)
        return {'size':size,'ok':fits,'peak_gib':round(peak/1024**3,3),
                'reason':'pass' if fits else 'insufficient headroom'}
    except (RuntimeError,MemoryError) as e:
        if 'out of memory' not in str(e).lower():raise
        return {'size':size,'ok':False,'peak_gib':None,'reason':str(e)[:160]}
    finally:
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()


def select_batch(model,teacher,ds,stage,device,cfg,mean,std,opt,log,provider,slot):
    tr=cfg['training'];initial=int(tr['batch_initial'])
    choices=[int(v) for v in tr['batch_choices']]
    if initial not in choices:raise ValueError('Initial batch absent from choices')
    if device.type!='cuda':return min(16,initial),[{'reason':'CPU limited to 16','size':min(16,initial)}]
    if str(tr['batch_mode'])=='fixed':return initial,[]
    attempts=[]
    first=probe_size(model,teacher,ds,stage,initial,device,cfg,mean,std,opt,provider,slot)
    attempts.append(first);log('RESHIFT_BATCH_PROBE '+json.dumps(first))
    if first['ok']:
        chosen=initial
        larger=[s for s in choices if s>initial]
        if larger and first['peak_gib']<torch.cuda.get_device_properties(device).total_memory/1024**3*.58:
            r=probe_size(model,teacher,ds,stage,min(larger),device,cfg,mean,std,opt,provider,slot)
            attempts.append(r);log('RESHIFT_BATCH_PROBE '+json.dumps(r))
            if r['ok']:chosen=min(larger)
        return chosen,attempts
    for size in sorted([s for s in choices if s<initial],reverse=True):
        r=probe_size(model,teacher,ds,stage,size,device,cfg,mean,std,opt,provider,slot)
        attempts.append(r);log('RESHIFT_BATCH_PROBE '+json.dumps(r))
        if r['ok']:return size,attempts
    raise RuntimeError('No safe CUDA batch, sizes attempted: '+str(attempts))


def save_last(path,key,stage,epoch,model,opt,scaler,best,best_epoch,wait,batch):
    atomic_torch({'schema':'ReShift-stage','stage_key':key,'stage':stage,'epoch':epoch,
                  'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),
                  'best':best,'best_epoch':best_epoch,'wait':wait,'batch':batch,'rng':rng_state()},path)


def reuse_stage(model,layout,stage,log):
    """Load a COMPLETE stage (own or shared) if its stage key matches; else None."""
    d=layout.dir(stage);done=d/'COMPLETE.json'
    if not done.exists():return None
    report=json.loads(done.read_text())
    if report.get('stage_key')!=layout.key(stage):
        raise RuntimeError(f'Stage key mismatch in {d}; use a new run/shared root')
    if sha256(d/'best.pt')!=report['best_sha256']:raise RuntimeError(f'Modified finished checkpoint {d}')
    load_model_state(model,checkpoint_load(d/'best.pt')['model'],layout.is_shared(stage))
    log(f'RESHIFT_STAGE_REUSE stage={stage} dir={d}')
    return report


def train_stage(model,train_ds,val_ds,sel_indices,device,cfg,mean,std,layout,stage,log,
                provider=None,retrieval_train_ds=None):
    stages=stages_for(cfg)
    if stage not in stages:raise ValueError(stage)
    idx=stages.index(stage);prev=stages[idx-1] if idx else None
    has_graph=(bool(cfg['model']['use_road_graph']) or bool(cfg['model']['use_cf_graph']))
    model.to(device)
    report=reuse_stage(model,layout,stage,log)
    if report is not None:
        model.set_stage(stage,use_graph=has_graph);return report
    key=layout.key(stage);stage_path=layout.dir(stage);stage_path.mkdir(parents=True,exist_ok=True)
    best_path=stage_path/'best.pt';last_path=stage_path/'last.pt';done=stage_path/'COMPLETE.json'
    lock=layout.acquire(stage)
    try:
        if prev and not last_path.exists():
            pd_=layout.dir(prev)
            if not (pd_/'COMPLETE.json').exists():raise RuntimeError('Previous stage not complete: '+prev)
            load_model_state(model,checkpoint_load(pd_/'best.pt')['model'],layout.is_shared(prev))
        model.set_stage(stage,use_graph=has_graph)
        model.to(device)
        retrieval_stage=stage in RETRIEVAL_STAGES
        data_ds=retrieval_train_ds if (retrieval_stage and retrieval_train_ds is not None) else train_ds
        if uses_evidence(model,stage):
            if provider is None:raise RuntimeError('retrieval stage without EvidenceProvider')
            prepare_train_banks(provider,data_ds,cfg,log)
            prepare_eval_banks(provider,val_ds,'val',cfg)
        slot_of=(provider.slot_for_epoch if (provider is not None and retrieval_stage) else (lambda ep:ep))
        teacher=PairedTeacher(model.stla).to(device)
        if stage=='anchor_cf' and getattr(model,'dose_enabled',False):
            teacher.dose_levels=tuple(model.cf.dose_levels[:-1])
            log('RESHIFT_DOSE_LADDER teacher_rungs='+json.dumps(list(teacher.dose_levels))
                +' student_grid='+json.dumps(list(model.cf.dose_levels)))
        normal_partition=(normal_calibration_partition(train_ds,cfg) if stage=='normal' else None)
        opt=stage_optimizer(model,stage,cfg)
        use_amp=bool(cfg['training']['amp']) and device.type=='cuda'
        scaler=torch.amp.GradScaler('cuda',enabled=use_amp)
        epochs=int(cfg['training']['stage_epochs'][stage]);patience=int(cfg['training']['patience'])
        batch=None;start=1;wait=0;best=float('inf');best_epoch=0
        ev_kw=dict(provider=provider,split='val')
        if last_path.exists():
            ck=checkpoint_load(last_path)
            if ck['stage_key']!=key or ck['stage']!=stage:raise RuntimeError('Stage resume identity mismatch')
            model.load_state_dict(ck['model'],strict=True)
            opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
            restore_rng(ck['rng'])
            start=int(ck['epoch'])+1;best=float(ck['best']);best_epoch=int(ck['best_epoch'])
            wait=int(ck['wait']);batch=int(ck['batch'])
            log(f'RESHIFT_RESUME stage={stage} epoch={start} batch={batch}')
        else:
            baseline=evaluate(model,val_ds,device,mean,std,int(cfg['training']['eval_batch_size']),
                              sel_indices,mode=stage,**ev_kw)
            if normal_partition is not None:
                baseline['train_only_normal_proxy_mae']=normal_calibration_score(
                    model,teacher,train_ds,normal_partition[1],device,std,int(cfg['training']['eval_batch_size']))
                baseline['selection_metric']=baseline['train_only_normal_proxy_mae']
            best=float(baseline['selection_metric'])
            atomic_torch({'stage_key':key,'stage':stage,'epoch':0,'model':model.state_dict(),
                          'selection':baseline},best_path)
            log(f'RESHIFT_STAGE_INITIAL stage={stage} selection={best:.6f}')
        if batch is None:
            data_ds.set_epoch(slot_of(start))
            batch,attempts=select_batch(model,teacher,data_ds,stage,device,cfg,mean,std,opt,log,
                                        provider,slot_of(start))
            atomic_json({'selected_batch':batch,'attempts':attempts},stage_path/'batch_probe.json')
        if not last_path.exists():save_last(last_path,key,stage,0,model,opt,scaler,best,best_epoch,wait,batch)
        target_effective=max(int(cfg['training']['effective_batch']),batch)
        for ep in range(start,epochs+1):
            slot=slot_of(ep)
            while True:
                data_ds.set_epoch(slot)
                try:
                    t0=time.perf_counter();model.train()
                    training_data=(Subset(data_ds,normal_partition[0]) if normal_partition is not None else data_ds)
                    loader=DataLoader(training_data,batch_size=batch,num_workers=int(cfg['training']['worker_count']),
                                      shuffle=True,drop_last=False,pin_memory=device.type=='cuda')
                    seen=0;sum_mae=0.;sums={}
                    opt.zero_grad(set_to_none=True)
                    accum=math.ceil(target_effective/batch)
                    for i,raw in enumerate(loader,1):
                        b=to_device(raw,device)
                        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=use_amp):
                            ev=_evidence(provider,model,stage,b,slot)
                            out=training_forward(model,b,stage,ev)
                            alt=training_contrast_forward(model,b,stage,cfg)
                            with torch.no_grad():targets=teacher(b,out['h0'],out['backbone'])
                            loss,details=(normal_anchor_loss(out,b,targets,stage,mean,std,cfg,other_views=alt)
                                          if model.prediction_mode=='normal_anchored' else
                                          compute_loss(out,b,targets,stage,mean,std,cfg,contrast=alt))
                        if not torch.isfinite(loss):raise RuntimeError('Nonfinite ReShift training loss')
                        in_group=((i-1)//accum)*accum
                        n_group=min(accum,len(loader)-in_group)
                        value=loss/n_group
                        if use_amp:scaler.scale(value).backward()
                        else:value.backward()
                        if i%accum==0 or i==len(loader):
                            if use_amp:scaler.unscale_(opt)
                            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                     float(cfg['training']['gradient_clip']))
                            if use_amp:scaler.step(opt);scaler.update()
                            else:opt.step()
                            opt.zero_grad(set_to_none=True)
                        num=int(b['future_valid'].sum());seen+=num
                        sum_mae+=float(details['mae'])*num
                        for dn,dv in details.items():
                            if dn=='mae':continue
                            sums[dn]=sums.get(dn,0.)+float(dv.detach())*num
                    break
                except (RuntimeError,MemoryError) as e:
                    if device.type!='cuda' or 'out of memory' not in str(e).lower():raise
                    lower=sorted([int(x) for x in cfg['training']['batch_choices'] if int(x)<batch],reverse=True)
                    if not lower:raise RuntimeError('Out of memory at smallest configured batch') from e
                    log(f'RESHIFT_RUNTIME_OOM stage={stage} epoch={ep} previous_batch={batch} next_batch={lower[0]}')
                    opt.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
                    ck=checkpoint_load(last_path)
                    model.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer'])
                    scaler.load_state_dict(ck['scaler']);restore_rng(ck['rng'])
                    batch=lower[0]
            scores=evaluate(model,val_ds,device,mean,std,int(cfg['training']['eval_batch_size']),
                            sel_indices,mode=stage,**ev_kw)
            if normal_partition is not None:
                scores['train_only_normal_proxy_mae']=normal_calibration_score(
                    model,teacher,train_ds,normal_partition[1],device,std,int(cfg['training']['eval_batch_size']))
                scores['selection_metric']=scores['train_only_normal_proxy_mae']
            metric=float(scores['selection_metric'])
            if metric<best-1e-7:
                best=metric;best_epoch=ep;wait=0
                atomic_torch({'stage_key':key,'stage':stage,'epoch':ep,'selection':scores,
                              'model':model.state_dict()},best_path)
            else:wait+=1
            peak=torch.cuda.max_memory_allocated(device)/1024**3 if device.type=='cuda' else 0
            log(f'RESHIFT_EPOCH stage={stage} epoch={ep}/{epochs} slot={slot} batch={batch} '
                f'train_mae_raw={sum_mae/max(seen,1):.6f} selection_macro_mae={metric:.6f} best={best:.6f} '
                f'loss_breakdown={json.dumps({k:round(v/max(seen,1),6) for k,v in sums.items()},sort_keys=True)} '
                f'selection_per_severity={json.dumps({k:round(v["final"]["mae"],5) for k,v in scores["per_severity"].items()})} '
                f'queries_per_second={len(data_ds)/max(1,time.perf_counter()-t0):.2f} peak_gib={peak:.3f} wait={wait}')
            save_last(last_path,key,stage,ep,model,opt,scaler,best,best_epoch,wait,batch)
            if wait>=patience:break
        report={'stage_key':key,'stage':stage,'best_epoch':best_epoch,'best_selection_macro_mae':best,
                'best_sha256':sha256(best_path)}
        atomic_json(report,done)
        model.load_state_dict(checkpoint_load(best_path)['model'],strict=True)
        model.set_stage(stage,use_graph=has_graph)
        log(f'RESHIFT_STAGE_PASS stage={stage} best_epoch={best_epoch} best_macro_mae={best:.6f}')
        return report
    finally:
        if lock is not None:lock.unlink(missing_ok=True)


# ------------------------------------------------------------------ banks
def query_desc(ds,split,cfg,severity=None,slot=None):
    return {'split':split,'severity':severity,'slot':slot,'rows':len(ds),
            'seed':int(cfg['experiment']['seed']),
            'kinds':list(cfg['training'].get('failure_kinds') or []),
            'manifest_sha256':sha256(Path(ds.raw.root)/f'{ds.raw.split}_manifest.csv')}


def prepare_train_banks(provider,ds,cfg,log):
    prior=ds.epoch
    try:
        for s in range(1,provider.slots+1):
            ds.set_epoch(s)
            provider.ensure(provider.train_key(s),ds,query_desc(ds,'train',cfg,slot=s),train_query=True)
    finally:
        ds.set_epoch(prior)


def prepare_eval_banks(provider,datasets,split,cfg,wrong=False):
    for sev,ds in datasets.items():
        provider.ensure(f'{split}_s{sev}',ds,query_desc(ds,split,cfg,severity=int(sev)),wrong=wrong)
