"""Reusable v9 training with chronological selection, checkpoint validation, profiling."""
from __future__ import annotations
import copy,json,os,time,random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
from .prepare import V9EvidenceDataset,V9PairedDataset
from .model import TCRRV9,train_loss,masked_mae,paired_cross_failure_loss,V9_VERSION
from ..v5.calibration import chronological_partitions


def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def save_atomic(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial.pt');torch.save(payload,tmp);os.replace(tmp,path)

def loader(ds,batch,shuffle,workers,seed=10):
    g=torch.Generator();g.manual_seed(seed)
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=workers,
                      pin_memory=torch.cuda.is_available(),persistent_workers=workers>0,
                      prefetch_factor=2 if workers>0 else None,generator=g,drop_last=False)

def to_device(batch,device):
    return {key:(x.to(device,non_blocking=True) if isinstance(x,torch.Tensor) else x)
            for key,x in batch.items()}

def one_epoch(model,data,optimizer,scaler,device,mean,std,stage,cfg,step_offset=0):
    training=optimizer is not None;model.train(training)
    total=0.;base_total=0.;normal_total=0.;valid_total=0;examples=0;t0=time.perf_counter()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    for step,raw in enumerate(data):
        if training:
            # Clear before every optimizer step: otherwise gradients accumulate across batches.
            optimizer.zero_grad(set_to_none=True)
        b=to_device(raw,device)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type,dtype=torch.float16,
                                enabled=device.type=='cuda' and cfg['amp']):
                output=model(base=b['base'],context=b['context'],x=b['x'],mask=b['mask'],
                             normal=b['normal'],cf=b['cf'],score=b['score'],valid=b['valid'],
                             match=b['match'],enable_cf=stage!='normal',
                             rank_mode='learned' if stage=='utility' else 'uniform')
            with torch.autocast(device_type=device.type,enabled=False):
                if stage=='normal':
                    loss=masked_mae(output['prediction'].float(),b['y_norm'].float(),b['future_valid'])
                    stats={'forecast':loss.detach()}
                else:
                    loss,stats=train_loss(output,b['y_norm'],b['future_valid'],b['valid'],
                            cfg['utility_weight'] if stage=='utility' else 0.,
                            cfg['ranking_weight'] if stage=='utility' else 0.,
                            teacher_cf=b.get('teacher_cf_training_only') if training else None,
                            full_info_weight=cfg.get('full_info_weight',0.) if stage=='utility' and training else 0.,
                            teacher_alignment_weight=cfg.get('teacher_alignment_weight',0.) if stage=='utility' and training else 0.)
            if training:
                if scaler.is_enabled():scaler.scale(loss).backward();scaler.unscale_(optimizer)
                else:loss.backward()
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg['clip'])
                if scaler.is_enabled():scaler.step(optimizer);scaler.update()
                else:optimizer.step()
        with torch.no_grad():
            w=b['future_valid'].float();v=w.sum().item();valid_total+=v
            base_total+=((b['base']-b['y_norm']).abs()*w).sum().item()
            normal_total+=((output['normal_prediction']-b['y_norm']).abs()*w).sum().item()
            total+=((output['prediction']-b['y_norm']).abs()*w).sum().item()
            examples+=len(b['base'])
    elapsed=time.perf_counter()-t0
    record=dict(stage=stage,rows=examples,seconds=elapsed,views_per_sec=examples/max(elapsed,1e-6),
                mae_norm=total/max(valid_total,1),base_mae_norm=base_total/max(valid_total,1),
                normal_mae_norm=normal_total/max(valid_total,1),
                cf_gain_norm=(normal_total-total)/max(valid_total,1))
    # MAE in raw units equals normalized MAE multiplied by fixed normalization std.
    record.update(mae_raw=record['mae_norm']*std,base_mae_raw=record['base_mae_norm']*std,
                  normal_mae_raw=record['normal_mae_norm']*std,
                  cf_gain_raw=record['cf_gain_norm']*std)
    if device.type=='cuda':
        record['gpu_alloc_gib']=torch.cuda.max_memory_allocated(device)/2**30
        record['gpu_reserved_gib']=torch.cuda.max_memory_reserved(device)/2**30
    return record

def one_epoch_cross(model,data,optimizer,scaler,device,mean,std,cfg,reference='official'):
    """Alternate which mask is the no-gradient reference; require real ID matches."""
    if reference not in ('official','paired'):raise ValueError('Invalid reference view')
    model.train(True);total=0.;count=0;t0=time.perf_counter();matched=0;possible=0
    for raw in data:
        other='paired' if reference=='official' else 'official'
        a=to_device(raw[reference],device);b=to_device(raw[other],device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            oa=model(base=a['base'],context=a['context'],x=a['x'],mask=a['mask'],
                    normal=a['normal'],cf=a['cf'],score=a['score'],valid=a['valid'],match=a['match'])
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=cfg['amp'] and device.type=='cuda'):
            ob=model(base=b['base'],context=b['context'],x=b['x'],mask=b['mask'],
                    normal=b['normal'],cf=b['cf'],score=b['score'],valid=b['valid'],match=b['match'])
        with torch.autocast(device_type=device.type,enabled=False):
            loss,st=train_loss(ob,b['y_norm'],b['future_valid'],b['valid'],
                               cfg['utility_weight'],cfg['ranking_weight'],
                teacher_cf=b.get('teacher_cf_training_only'),
                full_info_weight=cfg.get('full_info_weight',0.),
                teacher_alignment_weight=cfg.get('teacher_alignment_weight',0.))
            cross,alignment=paired_cross_failure_loss(oa,ob,a['candidate_ids'],b['candidate_ids'],
                  a['y_norm'],b['y_norm'],a['valid'],b['valid'],b['future_valid'],return_stats=True)
            loss=loss+cfg['cross_weight']*cross
        matched+=alignment['matched_candidates'];possible+=alignment['reference_valid_candidates']
        if scaler.is_enabled():scaler.scale(loss).backward();scaler.unscale_(optimizer)
        else:loss.backward()
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg['clip'])
        if scaler.is_enabled():scaler.step(optimizer);scaler.update()
        else:optimizer.step()
        count+=len(b['base']);total+=float(loss.detach())*len(b['base'])
    if matched==0:
        raise RuntimeError('No common valid candidate IDs across paired failures: Cross-Failure supervision is inactive')
    return dict(rows=count,seconds=time.perf_counter()-t0,total_loss=total/max(count,1),
                matching_rate=matched/max(possible,1),matched_candidates=matched,
                reference_valid_candidates=possible,reference_view=reference)

def run_training(cache,root,graph,context_dim,mean,std,output,seed,cfg,smoke=False,resume=None):
    set_seed(seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=TCRRV9(graph,context_dim,hidden=cfg['hidden'],bound=cfg['bound'],
                 blocks=cfg['blocks']).to(device)
    train=V9EvidenceDataset(cache,'train',(0,1),root=root)
    val=V9EvidenceDataset(cache,'val',(0,),root=root)
    if smoke:
        train=Subset(train,list(range(min(12,len(train)))))
        val=Subset(val,list(range(min(8,len(val)))))
    else:
        groups=chronological_partitions(val.ds.window_starts[:len(val.ds)],12,12,smoke=False)
        val=Subset(val,groups['selection'])
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    stages=[('normal',cfg['normal_epochs']),('transfer',cfg['transfer_epochs']),
            ('utility',cfg['utility_epochs'])]
    if cfg.get('cross_epochs',0)>0:stages.append(('cross',cfg['cross_epochs']))
    if smoke:stages=[(stage,1) for stage,_ in stages]
    if not stages or any(n<1 for _,n in stages):raise ValueError('Each requested training stage needs >=1 epoch')
    history=[];epoch_total=0;start_stage=0;resume_payload=None
    if resume:
        resume=Path(resume).resolve()
        if resume.parent!=out.resolve():
            raise ValueError('Resume checkpoint must be inside the same v9 run directory')
        ck=torch.load(resume,map_location='cpu',weights_only=False)
        if (ck.get('version')!=V9_VERSION or ck.get('seed')!=seed or
            ck.get('context_dim')!=context_dim or ck.get('cfg')!=cfg or
            ck.get('smoke')!=bool(smoke)):
            raise RuntimeError('Incompatible v9 resume checkpoint/configuration')
        model.load_state_dict(ck['model'],strict=True)
        history=list(ck['history']);start_stage=int(ck['stage_index'])
        epoch_total=int(ck['epoch_total']);resume_payload=ck
        if start_stage>=len(stages):raise RuntimeError('Stage beyond planned schedule')
        # Restore RNG states: check epoch sampler also seeds from epoch_total.
        random.setstate(ck['rng_python']);np.random.set_state(ck['rng_numpy'])
        torch.set_rng_state(ck['rng_torch'])
        if device.type=='cuda' and ck.get('rng_cuda') is not None:
            torch.cuda.set_rng_state_all(ck['rng_cuda'])
        print(f'V9_RESUME_PASS stage={ck["stage"]} epoch={ck["stage_epoch"]} '
              f'completed={epoch_total}',flush=True)
    for si,(stage,epochs) in enumerate(stages):
        if si<start_stage:continue
        for name,p in model.named_parameters():
            if name.startswith('graph.'):p.requires_grad_(False)
            elif stage=='normal':
                p.requires_grad_(name.startswith(('query.','match.','normal_fusion.','normal_scale.','none_normal')))
            elif stage=='transfer':
                p.requires_grad_(name.startswith(('cf_local.','cf_fusion.','amplitude.','temporal.')))
            else:
                # Freeze the normal anchor AND transferred CF amplitude. This is
                # a controlled utility-selection experiment, not joint finetuning.
                p.requires_grad_(name.startswith(('utility.','fullinfo.')) or name == 'none')
        resuming_same=(resume_payload is not None and si==start_stage)
        if si>0 and not resuming_same:
            prev=stages[si-1][0];p=out/f'best_{prev}.pt'
            if not p.is_file():raise FileNotFoundError(p)
            model.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['model'],strict=True)
        params=[p for p in model.parameters() if p.requires_grad]
        opt=torch.optim.AdamW(params,lr=cfg['lr']*(.5 if stage in ('utility','cross') else 1.),
                              weight_decay=1e-4)
        scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and cfg['amp'])
        best=float('inf');patience=0;first_ep=1
        if resuming_same:
            first_ep=int(resume_payload['stage_epoch'])+1
            best=float(resume_payload['stage_best'])
            patience=int(resume_payload['patience'])
            opt.load_state_dict(resume_payload['optimizer_state'])
            scaler.load_state_dict(resume_payload['scaler_state'])
        for ep in range(first_ep,epochs+1):
            epoch_total+=1
            ds=V9PairedDataset(cache,root) if stage=='cross' else train
            tl=loader(ds,cfg['batch_size'] if stage!='cross' else max(1,cfg['batch_size']//2),
                      True,cfg['workers'],seed+epoch_total)
            vl=loader(val,cfg['eval_batch_size'],False,0,seed)
            a=(one_epoch_cross(model,tl,opt,scaler,device,mean,std,cfg,
                               reference='official' if ep%2 else 'paired') if stage=='cross'
               else one_epoch(model,tl,opt,scaler,device,mean,std,stage,cfg))
            with torch.no_grad():
                b=one_epoch(model,vl,None,scaler,device,mean,std,
                            'utility' if stage=='cross' else stage,cfg)
            record={'epoch_total':epoch_total,'stage_epoch':ep,'stage':stage,'train':a,'val':b}
            history.append(record)
            print('V9_EPOCH '+json.dumps(record),flush=True)
            if b['mae_raw']<best-1e-6:
                best=b['mae_raw'];patience=0;improved=True
            else:
                patience+=1;improved=False
            (out/'history.json').write_text(json.dumps(history,indent=2))
            core=dict(version=V9_VERSION,seed=seed,context_dim=context_dim,
                      stage=stage,stage_index=si,stage_epoch=ep,
                      epoch_total=epoch_total,stage_best=best,patience=patience,
                      model=model.state_dict(),optimizer_state=opt.state_dict(),
                      scaler_state=scaler.state_dict(),
                      rng_python=random.getstate(),rng_numpy=np.random.get_state(),
                      rng_torch=torch.get_rng_state(),
                      rng_cuda=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                      history=history,cfg=cfg,graph_physical=graph.physical_available,smoke=bool(smoke))
            save_atomic(out/'last.pt',core)
            if improved:
                save_atomic(out/f'best_{stage}.pt',core)
            if patience>=cfg['patience'] and not smoke:break
        resume_payload=None
    return out/f'best_{stages[-1][0]}.pt'
