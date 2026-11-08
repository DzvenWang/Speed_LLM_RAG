"""Five-stage v9.2 optimization: Normal, Alignment, Utility, Joint, Paired.

Every CF phase freezes the complete Normal prediction path. Checkpoint identity,
optimizer/sampler and RNG states are resumable at epoch boundaries. Checkpoint
selection uses chronological validation selection only, never diagnostic/check.
"""
from __future__ import annotations
import json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset
from ..v9.training import set_seed,loader,to_device,save_atomic
from ..v9.prepare import V9EvidenceDataset as _V91Dataset
from ..v5.calibration import chronological_partitions
from ..v9.model import cross_failure_pairing,paired_cross_failure_loss,utility_targets
from .prepare import V9EvidenceDataset,V9PairedDataset
from .model import TCRRV92
from .utility import cf_training_loss
from . import VERSION

INPUTS=('base','context','x','mask','normal','cf','score','valid','match')

def forward_batch(model,b,**controls):return model(**{key:b[key] for key in INPUTS},**controls)

def configure_stage(model,stage):
    normal_names=('query.','match.','normal_fusion.','normal_scale.','none_normal')
    normal_train={'normal'}
    if stage not in ('normal','align','utility','joint','paired'):
        raise ValueError(f'Unknown stage {stage}')
    for name,p in model.named_parameters():
        if stage in normal_train:
            learn=name.startswith('normal_core.') and name[len('normal_core.'):].startswith(normal_names)
        elif name.startswith('normal_core.'):
            learn=False
        elif stage=='align':
            learn=name.startswith(('temporal_cf.','cf_graph.','cf_fusion.','alignment.','null_head.'))
        elif stage=='utility':
            learn=name.startswith(('utility.','null_head.','full_info_head.')) or name=='recoverability_scale'
        else:
            learn=name.startswith(('temporal_cf.','cf_graph.','cf_fusion.','alignment.','utility.','null_head.','full_info_head.')) or name=='recoverability_scale'
        p.requires_grad_(bool(learn))
    if stage!='normal' and any(p.requires_grad for p in model.normal_core.parameters()):
        raise AssertionError('Normal branch MUST be frozen in CF training')
    return [name for name,p in model.named_parameters() if p.requires_grad]


def _controls(stage,cfg):
    return dict(enable_cf=stage!='normal',alignment_mode=cfg['alignment_mode'],
                graph_mode=cfg['graph_mode'],temporal_mode=cfg['temporal_mode'],
                rank_mode='uniform' if stage=='align' else cfg.get('utility_mode','learned'),null_mode=cfg['null_mode'])


def _gradient_norms(model):
    grouped={}
    for name,p in model.named_parameters():
        if p.grad is None:continue
        head=name.split('.')[0] if not name.startswith('normal_core.') else 'normal_core'
        grouped[head]=grouped.get(head,0.)+float(p.grad.detach().float().square().sum())
    return {k:v**.5 for k,v in grouped.items()}


def one_epoch(model,data,optimizer,scaler,device,std,stage,cfg):
    train=optimizer is not None;model.train(train)
    # No train-time changes to fixed graph or Normal BN (the latter has no BN).
    total=0.;normal_total=0.;base_total=0.;denom=0.;num=0
    activation=negative=zero_alpha=saturation=0.;examples=0
    norms={};start=time.perf_counter()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    for raw in data:
        b=to_device(raw,device)
        if train:optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda' and cfg['amp']):
                out=forward_batch(model,b,**_controls(stage,cfg))
            with torch.autocast(device_type=device.type,enabled=False):
                loss,parts=cf_training_loss(out,b['y_norm'],b['future_valid'],b['valid'],cfg,stage,
                   teacher_cf=b.get('teacher_cf_training_only') if train else None,
                   raw_cf=b['cf'] if train else None)
            if train:
                if scaler.is_enabled():scaler.scale(loss).backward();scaler.unscale_(optimizer)
                else:loss.backward()
                grad=_gradient_norms(model)
                for name,value in grad.items():norms[name]=norms.get(name,0.)+value
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg['clip'])
                if scaler.is_enabled():scaler.step(optimizer);scaler.update()
                else:optimizer.step()
        with torch.no_grad():
            v=b['future_valid'].bool(); y=torch.where(v,b['y_norm'].float(),torch.zeros_like(b['y_norm'].float()))
            nerror=torch.where(v,(out['normal_prediction'].float()-y).abs(),torch.zeros_like(y))
            perror=torch.where(v,(out['prediction'].float()-y).abs(),torch.zeros_like(y))
            berror=torch.where(v,(out['base'].float()-y).abs(),torch.zeros_like(y))
            total+=float(perror.sum());normal_total+=float(nerror.sum());base_total+=float(berror.sum());denom+=float(v.sum())
            examples+=len(y);num+=1
            if 'alignment' in out:
                alpha=out['alignment'].float();negative+=float((alpha<-.05).float().mean());
                zero_alpha+=float((alpha.abs()<.05).float().mean());
                saturation+=float((alpha.abs()>model.max_align*.95).float().mean());
                activation+=float((1-out['cf_weights'][:,-1]).float().mean())
    elapsed=time.perf_counter()-start
    outrec=dict(stage=stage,rows=examples,seconds=elapsed,views_per_sec=examples/max(elapsed,1e-9),
         mae_norm=total/max(denom,1),base_mae_norm=base_total/max(denom,1),
         normal_mae_norm=normal_total/max(denom,1),cf_gain_norm=(normal_total-total)/max(denom,1),
         mae_raw=std*total/max(denom,1),normal_mae_raw=std*normal_total/max(denom,1),
         cf_gain_raw=std*(normal_total-total)/max(denom,1),
         active_cf_weight=activation/max(num,1),negative_alpha=negative/max(num,1),
         near_zero_alpha=zero_alpha/max(num,1),alpha_saturation=saturation/max(num,1),
         gradient_norm={k:v/max(num,1) for k,v in norms.items()})
    if device.type=='cuda':
        outrec['gpu_alloc_gib']=torch.cuda.max_memory_allocated(device)/2**30
        outrec['gpu_reserved_gib']=torch.cuda.max_memory_reserved(device)/2**30
    return outrec


def one_epoch_paired(model,data,optimizer,scaler,device,std,cfg,epoch):
    """Paired utility differences plus same-query matched/mismatched response tests."""
    model.train(True);start=time.perf_counter();n=0;weight_loss=0.;match_count=0;total_ref=0
    gradient_sums={}
    for raw in data:
        refkey='official' if epoch%2 else 'paired';other='paired' if epoch%2 else 'official'
        a=to_device(raw[refkey],device);b=to_device(raw[other],device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():oa=forward_batch(model,a,**_controls('paired',cfg))
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda' and cfg['amp']):
            ob=forward_batch(model,b,**_controls('paired',cfg))
        with torch.autocast(device_type=device.type,enabled=False):
            loss,parts=cf_training_loss(ob,b['y_norm'],b['future_valid'],b['valid'],cfg,'paired',
                            teacher_cf=b.get('teacher_cf_training_only'),raw_cf=b['cf'])
            cross,stat=paired_cross_failure_loss(oa,ob,a['candidate_ids'],b['candidate_ids'],
                   a['y_norm'],b['y_norm'],a['valid'],b['valid'],b['future_valid'],return_stats=True)
            match_count+=stat['matched_candidates'];total_ref+=stat['reference_valid_candidates']
            loss=loss+cfg['cross_weight']*cross
            # For each candidate in B, compare its CF response to the SAME ID's
            # response under A. B's query, mask, Normal and Y remain unchanged.
            idx,pair=cross_failure_pairing(b['candidate_ids'],a['candidate_ids'],b['valid'],a['valid'])
            alt=a['cf'].gather(1,idx[:,:,None,None].expand_as(b['cf']))
            mismatch=torch.where(pair[:,:,None,None],alt,b['cf'])
            with torch.no_grad():
                wrong=forward_batch(model,{**b,'cf':mismatch},**_controls('paired',cfg))
                true_good=utility_targets(ob['base'],ob['normal_raw'],ob['candidate_cf'],b['y_norm'],
                       b['valid'],ob['bound'],model.blocks,b['future_valid'])
                true_wrong=utility_targets(wrong['base'],wrong['normal_raw'],wrong['candidate_cf'],b['y_norm'],
                       b['valid'],wrong['bound'],model.blocks,b['future_valid'])
                delta_truth=true_good-true_wrong
                delta_reference=wrong['cf_utility'].detach()
            fv=b['future_valid'].reshape(len(b['base']),model.blocks,-1,b['future_valid'].shape[-1]).any(2)
            effective=pair[:,:,None,None]&fv[:,None]
            d=nn.functional.smooth_l1_loss(ob['cf_utility'].float()-delta_reference.float(),
                                           delta_truth.float(),reduction='none')
            contrast=torch.where(effective,d,torch.zeros_like(d)).sum()/effective.sum().clamp_min(1)
            loss=loss+cfg['compat_weight']*contrast
        if scaler.is_enabled():scaler.scale(loss).backward();scaler.unscale_(optimizer)
        else:loss.backward()
        grad=_gradient_norms(model)
        for name,value in grad.items():gradient_sums[name]=gradient_sums.get(name,0.)+value
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg['clip'])
        if scaler.is_enabled():scaler.step(optimizer);scaler.update()
        else:optimizer.step()
        weight_loss+=float(loss.detach())*len(b['base']);n+=len(b['base'])
    if match_count==0:raise RuntimeError('Cross-failure stage had zero matching time-legal candidates')
    return dict(stage='paired',rows=n,seconds=time.perf_counter()-start,total_loss=weight_loss/max(n,1),
         matched_candidates=match_count,reference_valid_candidates=total_ref,
         matching_rate=match_count/max(total_ref,1),gradient_norm={k:v/max(1,n) for k,v in gradient_sums.items()})


def create_model(graph,context_dim,cfg):
    return TCRRV92(graph,context_dim,hidden=cfg['hidden'],cf_hidden=cfg.get('cf_hidden',16),
                  blocks=cfg['blocks'],bound=cfg['bound'],max_align=cfg.get('max_align',1.5),
                  temp=cfg.get('temperature',.2))


def run_training(cache,root,graph,context_dim,mean,std,output,seed,cfg,smoke=False,resume=None):
    set_seed(seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=create_model(graph,context_dim,cfg).to(device)
    train=V9EvidenceDataset(cache,'train',(0,1),root=root)
    valid=V9EvidenceDataset(cache,'val',(0,),root=root)
    if smoke:
        train=Subset(train,list(range(min(12,len(train)))))
        valid=Subset(valid,list(range(min(8,len(valid)))))
    else:
        split=chronological_partitions(valid.ds.window_starts[:len(valid.ds)],12,12)
        valid=Subset(valid,split['selection'])
    stages=[(name,int(cfg.get(name+'_epochs',0))) for name in ('normal','align','utility','joint','paired')]
    stages=[(name,ep) for name,ep in stages if ep>0]
    if not stages or stages[0][0]!='normal':raise ValueError('Training must start with Normal anchor')
    if smoke:stages=[(name,1) for name,_ in stages]
    folder=Path(output);folder.mkdir(parents=True,exist_ok=True)
    history=[];epoch_total=0;start_stage=0;resume_payload=None
    if resume:
        resume=Path(resume).resolve()
        if resume.parent!=folder.resolve():raise ValueError('Resume must refer to same run directory')
        ck=torch.load(resume,map_location='cpu',weights_only=False)
        if (ck['version']!=VERSION or ck['seed']!=seed or ck['cfg']!=cfg or
            ck['context_dim']!=context_dim or ck['smoke']!=bool(smoke)):
            raise RuntimeError('v9.2 resume metadata/configuration mismatch')
        model.load_state_dict(ck['model'],strict=True)
        history=list(ck['history']);start_stage=ck['stage_index'];epoch_total=ck['epoch_total'];resume_payload=ck
        random.setstate(ck['rng_python']);np.random.set_state(ck['rng_numpy']);torch.set_rng_state(ck['rng_torch'])
        if device.type=='cuda' and ck.get('rng_cuda') is not None:torch.cuda.set_rng_state_all(ck['rng_cuda'])
    # Ensure frozen Normal parameters never change after Stage 0 (including
    # optimizer weight decay or accidental stage-prefix selection).
    normal_reference=None
    for si,(stage,epochs) in enumerate(stages):
        if si<start_stage:continue
        trainable=configure_stage(model,stage)
        if si>0 and not (resume_payload is not None and si==start_stage):
            ck=torch.load(folder/f'best_{stages[si-1][0]}.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(ck['model'],strict=True)
        if si==1:normal_reference={key:p.detach().cpu().clone() for key,p in model.normal_core.named_parameters()}
        if si>1 and normal_reference is None:
            checkpoint=torch.load(folder/'best_normal.pt',map_location='cpu',weights_only=False)
            normal_reference={key:value.detach().cpu().clone() for key,value in
                   ((name,checkpoint['model']['normal_core.'+name]) for name,_ in model.normal_core.named_parameters())}
        params=[p for p in model.parameters() if p.requires_grad]
        if not params:raise RuntimeError('No trainable parameters for '+stage)
        lr=cfg['lr']*(.5 if stage in ('utility','joint','paired') else 1.)
        optimizer=torch.optim.AdamW(params,lr=lr,weight_decay=1e-4)
        scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and cfg['amp'])
        best=float('inf');patience=0;first_ep=1
        if resume_payload is not None and si==start_stage:
            first_ep=int(resume_payload['stage_epoch'])+1
            best=float(resume_payload['stage_best']);patience=int(resume_payload['patience'])
            optimizer.load_state_dict(resume_payload['optimizer_state'])
            scaler.load_state_dict(resume_payload['scaler_state'])
        for ep in range(first_ep,epochs+1):
            epoch_total+=1
            ds=V9PairedDataset(cache,root) if stage=='paired' else train
            batch=max(1,cfg['batch_size']//2) if stage=='paired' else cfg['batch_size']
            tl=loader(ds,batch,True,cfg['workers'],seed+epoch_total)
            vl=loader(valid,cfg['eval_batch_size'],False,0,seed)
            tr=(one_epoch_paired(model,tl,optimizer,scaler,device,std,cfg,ep) if stage=='paired'
                else one_epoch(model,tl,optimizer,scaler,device,std,stage,cfg))
            with torch.no_grad():vr=one_epoch(model,vl,None,scaler,device,std,stage,cfg)
            if normal_reference is not None:
                mismatched=[name for name,p in model.normal_core.named_parameters() if
                            not torch.equal(p.detach().cpu(),normal_reference[name])]
                if mismatched:raise RuntimeError('Frozen Normal anchor was modified: '+str(mismatched))
            record=dict(epoch_total=epoch_total,stage=stage,stage_epoch=ep,train=tr,val=vr,
                         trainable_parameters=trainable)
            history.append(record);print('V92_EPOCH '+json.dumps(record),flush=True)
            improved=vr['mae_raw']<best-1e-6
            if improved:best=vr['mae_raw'];patience=0
            else:patience+=1
            (folder/'history.json').write_text(json.dumps(history,indent=2))
            checkpoint=dict(version=VERSION,seed=seed,cfg=cfg,context_dim=context_dim,
                  stage=stage,stage_index=si,stage_epoch=ep,epoch_total=epoch_total,
                  stage_best=best,patience=patience,model=model.state_dict(),
                  optimizer_state=optimizer.state_dict(),scaler_state=scaler.state_dict(),
                  history=history,rng_python=random.getstate(),rng_numpy=np.random.get_state(),
                  rng_torch=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                  smoke=bool(smoke),graph_physical=graph.physical_available)
            save_atomic(folder/'last.pt',checkpoint)
            if improved:save_atomic(folder/f'best_{stage}.pt',checkpoint)
            if patience>=cfg['patience'] and not smoke:break
        resume_payload=None
    best_stage=min((name for name,_ in stages),key=lambda name:
                   float(torch.load(folder/f'best_{name}.pt',map_location='cpu',weights_only=False)['stage_best']))
    # IMPORTANT: full model selection only among CF stages; the Normal checkpoint
    # remains an explicit fallback if no CF candidate improves validation MAE.
    best_path=folder/f'best_{best_stage}.pt'
    (folder/'selected_checkpoint.json').write_text(json.dumps({'stage':best_stage,
          'path':str(best_path.resolve()),'selection_only':True},indent=2))
    return best_path
