"""Full staged trainer: CF -> Residual -> Joint(+Graph) -> Benefit Control.
Identity-locked resume, four-severity validation selection, no TEST before lock.
"""
from __future__ import annotations
import gc,json,math,time
import numpy as np
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
from torch.utils.data._utils.collate import default_collate
from corsta_v2.util import (atomic_json,atomic_torch,checkpoint_load,to_device,
                             rng_state,restore_rng,sha256)
from corsta_v2.counterfactual.teacher import PairedTeacher
from corsta_v2.training.losses import compute_loss
from corsta_v2.training.normal_anchor import normal_anchor_loss
from corsta_v2.training.evaluation import evaluate

CF_FIRST_STAGES=('cf','residual','joint','control')
NORMAL_ANCHORED_STAGES=('normal','anchor_cf','anchor_control')
STAGES=CF_FIRST_STAGES  # legacy v0.2.1 compatibility


def stages_for(cfg):
    return (NORMAL_ANCHORED_STAGES if cfg['model'].get('prediction_mode')=='normal_anchored'
            else CF_FIRST_STAGES)



def normal_calibration_partition(ds,config):
    """Time-embargoed *TRAIN* holdout, never complete VAL/TEST Teacher labels.

    The calibration windows are excluded from the Normal-stage optimizer data.
    """
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
    """Teacher-forced Normal error on disjoint TRAIN-only calibration data.

    Full teacher reference is used ONLY for this selection statistic, never in
    inference/VAL/TEST.  All episodes use epoch=0 masks to keep comparisons fixed.
    """
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
            # TRAIN-only perfect-CF proxy isolates full-view Normal residual.
            corrected=o['normal_prediction']+t['cf_target']
            valid=b['future_valid'].float()
            y=torch.where(valid.bool(),b['y_norm'].float(),torch.zeros_like(b['y_norm'].float()))
            err=(corrected-y).abs()*std
            total+=float((err*valid).sum());count+=float(valid.sum())
    finally:
        ds.set_epoch(prior)
    return total/max(count,1.)


def stage_mode(stage):return stage


def stage_optimizer(model,stage,cfg):
    tr=cfg['training'];groups=[]
    names={'normal':('failure','residual','graph_residual'),
           'anchor_cf':('cf','input_adapter','graph_cf'),
           'anchor_control':('cf',),
           'cf':('failure','input_adapter','cf'),
           'residual':('residual',),
           'joint':('failure','input_adapter','cf','residual','graph_cf','graph_residual'),
           'control':('controller',)}[stage]
    for name in names:
        module=getattr(model,name)
        params=[p for p in module.parameters() if p.requires_grad]
        if not params:continue
        lr=(tr['lr_cf'] if name in ('cf','failure','input_adapter') else
            tr['lr_residual'] if name=='residual' else
            tr['lr_graph'] if name.startswith('graph') else tr['lr_controller'])
        groups.append({'params':params,'lr':float(lr),'name':name})
    used={id(p) for grp in groups for p in grp['params']}
    missing=[n for n,p in model.named_parameters() if p.requires_grad and id(p) not in used]
    if missing:raise RuntimeError('Optimizer missing '+str(missing[:10]))
    if not groups:raise RuntimeError('No trainable parameters')
    return torch.optim.AdamW(groups,weight_decay=float(tr['weight_decay']))


def training_forward(model,b,stage):
    if model.prediction_mode=='normal_anchored':
        return model({k:b[k] for k in ('x','mask','tod','dow')},
                     enable_cf=(stage!='normal'),
                     enable_controller=(stage=='anchor_control'))
    return model({k:b[k] for k in ('x','mask','tod','dow')},
                 enable_cf=True,enable_residual=(stage!='cf'),
                 enable_controller=(stage=='control'))


def training_contrast_forward(model,b,stage,cfg):
    """Second masked view, NEVER a second future or a full-view Student input."""
    if not cfg['training'].get('multimask_error_contrast',False):return None
    if model.prediction_mode=='normal_anchored':
        if stage!='anchor_cf':return None
        if 'view_x' not in b or 'view_mask' not in b:
            raise RuntimeError('Normal-anchored K-view training requires view_x/view_mask')
        results=[]
        for k in range(b['view_x'].shape[1]):
            results.append(model({'x':b['view_x'][:,k],'mask':b['view_mask'][:,k],
                                  'tod':b['tod'],'dow':b['dow']},
                                 enable_cf=True,enable_controller=False))
        return results
    if stage not in ('cf','joint'):return None
    if 'contrast_x' not in b or 'contrast_mask' not in b:
        raise RuntimeError('Multimask error contrast enabled but TRAIN lacks contrast view')
    return model({'x':b['contrast_x'],'mask':b['contrast_mask'],
                  'tod':b['tod'],'dow':b['dow']},
                 enable_cf=True,enable_residual=False,
                 enable_controller=False)


def _make_probe_batch(dataset,size,device):
    item=dataset[0]
    # This uses the actual dataset keys and shapes; no synthetic shrinkage.
    batch=default_collate([item]*size)
    return to_device(batch,device)


def probe_size(model,teacher,dataset,stage,size,device,cfg,mean,std,optimizer):
    amp=bool(cfg['training']['amp']) and device.type=='cuda'
    if device.type=='cuda':torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
    try:
        b=_make_probe_batch(dataset,size,device)
        model.train()
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
            out=training_forward(model,b,stage)
            alt=training_contrast_forward(model,b,stage,cfg)
            with torch.no_grad(): t=teacher(b,out['h0'],out['backbone'])
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
        for name in ('b','out','alt','loss','t'):
            if name in locals():del locals()[name]
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()


def select_batch(model,teacher,ds,stage,device,cfg,mean,std,opt,log):
    tr=cfg['training'];initial=int(tr['batch_initial'])
    choices=[int(v) for v in tr['batch_choices']]
    if initial not in choices:raise ValueError('Initial batch absent from choices')
    if device.type!='cuda':return min(16,initial),[{'reason':'CPU limited to 16 (no VRAM)', 'size':min(16,initial)}]
    if str(tr['batch_mode'])=='fixed':return initial,[]
    attempts=[]
    first=probe_size(model,teacher,ds,stage,initial,device,cfg,mean,std,opt)
    attempts.append(first);log('CORSTA_V2_BATCH_PROBE '+json.dumps(first))
    if first['ok']:
        chosen=initial
        larger=[s for s in choices if s>initial]
        if larger and first['peak_gib'] < torch.cuda.get_device_properties(device).total_memory/1024**3*.58:
            r=probe_size(model,teacher,ds,stage,min(larger),device,cfg,mean,std,opt)
            attempts.append(r);log('CORSTA_V2_BATCH_PROBE '+json.dumps(r))
            if r['ok']:chosen=min(larger)
        return chosen,attempts
    for size in sorted([s for s in choices if s<initial],reverse=True):
        r=probe_size(model,teacher,ds,stage,size,device,cfg,mean,std,opt)
        attempts.append(r);log('CORSTA_V2_BATCH_PROBE '+json.dumps(r))
        if r['ok']:return size,attempts
    raise RuntimeError('No safe CUDA batch, sizes attempted: '+str(attempts))


def save_last(path,identity,stage,epoch,model,opt,scaler,best,best_epoch,wait,batch):
    atomic_torch({'schema':'CoRSTA-v2-stage','identity':identity,'stage':stage,'epoch':epoch,
                  'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),
                  'best':best,'best_epoch':best_epoch,'wait':wait,'batch':batch,
                  'rng':rng_state()},path)


def train_stage(model,train_ds,val_ds,sel_indices,device,cfg,mean,std,outdir,identity,stage,log):
    stages=stages_for(cfg)
    if stage not in stages:raise ValueError(stage)
    stage_index=stages.index(stage)
    stage_path=Path(outdir)/stage
    stage_path.mkdir(parents=True,exist_ok=True)
    done=stage_path/'COMPLETE.json'
    best_path=stage_path/'best.pt';last_path=stage_path/'last.pt'
    prev=stages[stage_index-1] if stage_index else None
    if done.exists():
        report=json.loads(done.read_text())
        if report.get('identity')!=identity:raise RuntimeError('Stage identity mismatch')
        if sha256(best_path)!=report['best_sha256']:raise RuntimeError('Modified finished stage checkpoint')
        model.load_state_dict(checkpoint_load(best_path)['model'],strict=True)
        log('CORSTA_V2_STAGE_REUSE stage='+stage)
        return report
    if prev and not last_path.exists():
        prevfile=Path(outdir)/prev/'best.pt'
        if not (Path(outdir)/prev/'COMPLETE.json').exists():raise RuntimeError('Previous stage not complete: '+prev)
        model.load_state_dict(checkpoint_load(prevfile)['model'],strict=True)
    has_graph=(bool(cfg['model']['use_road_graph']) or bool(cfg['model']['use_cf_graph']))
    model.set_stage(stage,use_graph=(has_graph if model.prediction_mode=='normal_anchored'
                                  else (has_graph and stage in ('joint','control'))))
    model.to(device)
    teacher=PairedTeacher(model.stla).to(device)
    normal_partition=(normal_calibration_partition(train_ds,cfg)
                      if stage=='normal' else None)
    if stage=='normal' and normal_partition is None:
        log('CORSTA_V022_NORMAL_CALIBRATION_FALLBACK: use deployable VAL normal MAE (smoke/short TRAIN)')
    if normal_partition is not None:
        log(f'CORSTA_V022_NORMAL_TRAIN_CALIBRATION train={len(normal_partition[0])} '
            f'calibration={len(normal_partition[1])} embargo={cfg["training"].get("normal_calibration_embargo",24)}')
    opt=stage_optimizer(model,stage,cfg)
    use_amp=bool(cfg['training']['amp']) and device.type=='cuda'
    scaler=torch.amp.GradScaler('cuda',enabled=use_amp)
    epochs=int(cfg['training']['stage_epochs'][stage]);patience=int(cfg['training']['patience'])
    batch=None;start=1;wait=0;best=float('inf');best_epoch=0
    if last_path.exists():
        ck=checkpoint_load(last_path)
        if ck['identity']!=identity or ck['stage']!=stage:raise RuntimeError('Stage resume identity mismatch')
        model.load_state_dict(ck['model'],strict=True)
        opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
        restore_rng(ck['rng'])
        start=int(ck['epoch'])+1;best=float(ck['best']);best_epoch=int(ck['best_epoch']);wait=int(ck['wait']);batch=int(ck['batch'])
        log(f'CORSTA_V2_RESUME stage={stage} epoch={start} batch={batch}')
    else:
        baseline=evaluate(model,val_ds,device,mean,std,int(cfg['training']['eval_batch_size']),sel_indices,mode=stage)
        if normal_partition is not None:
            baseline['train_only_normal_proxy_mae']=normal_calibration_score(
                model,teacher,train_ds,normal_partition[1],device,std,
                int(cfg['training']['eval_batch_size']))
            baseline['selection_metric']=baseline['train_only_normal_proxy_mae']
        best=float(baseline['selection_metric'])
        atomic_torch({'identity':identity,'stage':stage,'epoch':0,'model':model.state_dict(),
                      'selection':baseline},best_path)
        log(f'CORSTA_V2_STAGE_INITIAL stage={stage} selection={best:.6f}')
    if batch is None:
        train_ds.set_epoch(start)
        batch,attempts=select_batch(model,teacher,train_ds,stage,device,cfg,mean,std,opt,log)
        atomic_json({'selected_batch':batch,'attempts':attempts},stage_path/'batch_probe.json')
    if not last_path.exists():save_last(last_path,identity,stage,0,model,opt,scaler,best,best_epoch,wait,batch)
    target_effective=max(int(cfg['training']['effective_batch']),batch)
    for ep in range(start,epochs+1):
        while True:
            train_ds.set_epoch(ep)
            # Restore at the epoch boundary on CUDA OOM, never preserve partially updated params.
            try:
                start_time=time.perf_counter();model.train()
                training_data=(Subset(train_ds,normal_partition[0])
                               if normal_partition is not None else train_ds)
                train_loader=DataLoader(training_data,batch_size=batch,num_workers=int(cfg['training']['worker_count']),
                                        shuffle=True,drop_last=False,pin_memory=device.type=='cuda')
                optim_steps=0;seen=0;sum_mae=0.;train_detail_sums={}
                opt.zero_grad(set_to_none=True)
                accum=math.ceil(target_effective/batch)
                for i,raw in enumerate(train_loader,1):
                    b=to_device(raw,device)
                    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=use_amp):
                        out=training_forward(model,b,stage)
                        alt=training_contrast_forward(model,b,stage,cfg)
                        with torch.no_grad():targets=teacher(b,out['h0'],out['backbone'])
                        loss,details=(normal_anchor_loss(out,b,targets,stage,mean,std,cfg,
                                                         other_views=alt)
                                      if model.prediction_mode=='normal_anchored' else
                                      compute_loss(out,b,targets,stage,mean,std,cfg,contrast=alt))
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite CoRSTA training loss')
                    # Equal weight per micro-batch, including the short final group.
                    in_group=((i-1)//accum)*accum
                    n_group=min(accum,len(train_loader)-in_group)
                    value=loss/n_group
                    if use_amp:scaler.scale(value).backward()
                    else:value.backward()
                    if i%accum==0 or i==len(train_loader):
                        if use_amp:scaler.unscale_(opt)
                        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],float(cfg['training']['gradient_clip']))
                        if use_amp:scaler.step(opt);scaler.update()
                        else:opt.step()
                        opt.zero_grad(set_to_none=True);optim_steps+=1
                    num=int(b['future_valid'].sum());seen+=num
                    sum_mae+=float(details['mae'])*num
                    for detail_name,detail_value in details.items():
                        if detail_name=='mae':continue
                        train_detail_sums[detail_name]=train_detail_sums.get(detail_name,0.)+float(detail_value.detach())*num
                break
            except (RuntimeError,MemoryError) as e:
                if device.type!='cuda' or 'out of memory' not in str(e).lower():raise
                choices=sorted([int(x) for x in cfg['training']['batch_choices'] if int(x)<batch],reverse=True)
                if not choices:raise RuntimeError('Out of memory at smallest configured batch') from e
                lower=choices[0]
                log(f'CORSTA_V2_RUNTIME_OOM stage={stage} epoch={ep} previous_batch={batch} next_batch={lower}')
                opt.zero_grad(set_to_none=True)
                if 'train_loader' in locals():del train_loader
                if 'out' in locals():del out
                if 'b' in locals():del b
                gc.collect();torch.cuda.empty_cache()
                ck=checkpoint_load(last_path)
                model.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer'])
                scaler.load_state_dict(ck['scaler']);restore_rng(ck['rng'])
                batch=lower
        scores=evaluate(model,val_ds,device,mean,std,int(cfg['training']['eval_batch_size']),sel_indices,mode=stage)
        if normal_partition is not None:
            scores['train_only_normal_proxy_mae']=normal_calibration_score(
                model,teacher,train_ds,normal_partition[1],device,std,
                int(cfg['training']['eval_batch_size']))
            scores['selection_metric']=scores['train_only_normal_proxy_mae']
        metric=float(scores['selection_metric'])
        if metric < best - 1e-7:
            best=metric;best_epoch=ep;wait=0
            atomic_torch({'identity':identity,'stage':stage,'epoch':ep,
                          'selection':scores,'model':model.state_dict()},best_path)
        else:wait+=1
        peak=torch.cuda.max_memory_allocated(device)/1024**3 if device.type=='cuda' else 0
        epoch_loss_details={key:round(value/max(seen,1),6)
                            for key,value in train_detail_sums.items()}
        log(f'CORSTA_V022_EPOCH stage={stage} epoch={ep}/{epochs} batch={batch} accum={math.ceil(target_effective/batch)} '
            f'train_mae_raw={sum_mae/max(seen,1):.6f} selection_macro_mae={metric:.6f} best={best:.6f} '
            f'loss_breakdown={json.dumps(epoch_loss_details,sort_keys=True)} '
            f'selection_per_severity={json.dumps({k:round(v["final"]["mae"],5) for k,v in scores["per_severity"].items()})} '
            f'queries_per_second={len(train_ds)/max(1,time.perf_counter()-start_time):.2f} peak_gib={peak:.3f} wait={wait}')
        save_last(last_path,identity,stage,ep,model,opt,scaler,best,best_epoch,wait,batch)
        if wait>=patience:break
    report={'identity':identity,'stage':stage,'best_epoch':best_epoch,'best_selection_macro_mae':best,
            'best_sha256':sha256(best_path)}
    atomic_json(report,done)
    model.load_state_dict(checkpoint_load(best_path)['model'],strict=True)
    log(f'CORSTA_V2_STAGE_PASS stage={stage} best_epoch={best_epoch} best_macro_mae={best:.6f}')
    return report
