"""Stage-separated v9.3 trainer with frozen Normal anchor and chronological evaluation.

Stages: Residual -> CF response -> Utility -> CF joint. Train-only CF teacher.
"""
from __future__ import annotations
import copy,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
from ..v5.calibration import chronological_partitions
from .dataset import DirectViewDataset
from .model import TCRRV93,bounded
from .losses import stage_loss,mae,masked_mean
from .teacher import atomic_json

VERSION='9.3.0'
STAGES=('residual','cf_response','utility','cf_joint')


def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def to_device(x,device):
    if torch.is_tensor(x):return x.to(device,non_blocking=True)
    if isinstance(x,dict):return {k:to_device(v,device) for k,v in x.items()}
    return x


def stage_parameters(model,stage):
    ownership={
      'residual':('normal_query','residual_head'),
      'cf_response':('cf_mask_encoder','cf_adapter','graph','cf_response_head'),
      'utility':('cf_controller',),
      'cf_joint':('cf_mask_encoder','cf_adapter','graph','cf_response_head','cf_controller'),
    }
    if stage not in ownership:raise ValueError(stage)
    groups=ownership[stage]
    for n,p in model.named_parameters():p.requires_grad_(n.split('.')[0] in groups)
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    if not names:raise RuntimeError('Empty trainable stage')
    return names


def normal_fingerprint(model):
    h=hashlib.sha256()
    for root in (model.normal_query,model.residual_head):
        for key,v in root.state_dict().items():
            h.update(key.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def normal_forward(model,student):
    r,_=model.normal(student)
    b=student['base'].float()
    normal=b+bounded(r,model.bound)
    return dict(base=b,normal_raw=r,normal_prediction=normal,prediction=normal)


def make_loader(ds,batch_size,workers,shuffle,seed):
    gen=torch.Generator().manual_seed(seed)
    return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,num_workers=workers,
                      pin_memory=torch.cuda.is_available(),persistent_workers=bool(workers),
                      generator=gen,drop_last=False)


def _epoch(model,loader,stage,device,opt,scaler,conf,training,paired=False):
    model.train(training)
    # Frozen normal branch cannot drift via dropout/batchnorm or running stats.
    if stage!='residual':model.normal_query.eval();model.residual_head.eval()
    total=0.;norm=0.;base=0.;denom=0;losses=0.;steps=0;gain_positive=0.;gain_negative=0.
    start=time.perf_counter();num_grad=0
    for raw in loader:
        raw=to_device(raw,device)
        a=raw['a'] if paired else raw
        other=raw['b'] if paired else None
        if training:opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type,dtype=torch.float16,
                                enabled=scaler.is_enabled() if scaler else False):
                output=normal_forward(model,a['student']) if stage=='residual' else model(a['student'])
                pair_output=model(other['student']) if other is not None else None
            with torch.autocast(device_type=device.type,enabled=False):
                pair_obj={'batch':other,'output':pair_output} if other is not None else None
                if (not training) and stage=='residual' and model.mode=='decomposed':
                    loss=mae(output['normal_prediction'],a['y'],a['future_valid']);parts={'forecast':float(loss.detach())}
                else:
                    loss,parts=stage_loss(stage,output,a,model.mode,model.bound,conf['loss'],paired=pair_obj)
            if not torch.isfinite(loss):raise RuntimeError(f'Non-finite {stage} loss')
            if training:
                if scaler and scaler.is_enabled():
                    scaler.scale(loss).backward();scaler.unscale_(opt)
                else:loss.backward()
                params=[p for p in model.parameters() if p.requires_grad]
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in params):
                    raise RuntimeError(f'Non-finite gradient during {stage}')
                num_grad+=sum(int(p.grad is not None and torch.count_nonzero(p.grad)>0) for p in params)
                nn.utils.clip_grad_norm_(params,float(conf['training']['clip']))
                if scaler and scaler.is_enabled():scaler.step(opt);scaler.update()
                else:opt.step()
        with torch.no_grad():
            valid=a['future_valid'].bool()
            y=a['y'].float()
            p=torch.where(valid,(output['prediction'].float()-y).abs(),torch.zeros_like(y))
            n=torch.where(valid,(output['normal_prediction'].float()-y).abs(),torch.zeros_like(y))
            b=torch.where(valid,(output['base'].float()-y).abs(),torch.zeros_like(y))
            total+=float(p.sum());norm+=float(n.sum());base+=float(b.sum());denom+=int(valid.sum())
            if 'cf_increment' in output:
                d=n-p;gain_positive+=int(((d>0)&valid).sum());gain_negative+=int(((d<0)&valid).sum())
            losses+=float(loss.detach());steps+=1
    if training and num_grad==0:raise RuntimeError(f'Zero effective gradients in stage {stage}')
    return dict(stage=stage,mae_norm=total/max(denom,1),normal_mae_norm=norm/max(denom,1),
                base_mae_norm=base/max(denom,1),cf_gain_norm=(norm-total)/max(denom,1),
                harm_rate=gain_negative/max(denom,1),benefit_rate=gain_positive/max(denom,1),
                loss=losses/max(steps,1),samples=denom,seconds=time.perf_counter()-start,
                effective_gradient_tensors=num_grad)


def atomic_save(path,payload):
    path=Path(path);tmp=path.with_suffix('.partial.pt')
    torch.save(payload,tmp);os.replace(tmp,path)


def rng_snapshot():
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def rng_restore(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'])
    if torch.cuda.is_available() and r['cuda'] is not None:torch.cuda.set_rng_state_all(r['cuda'])


def training_signature(teacher_root,config,mode,graph_mode,context_dim,nodes,seed,source_id):
    return dict(version=VERSION,teacher_source=source_id,mode=mode,graph=graph_mode,
                context_dim=context_dim,nodes=nodes,seed=int(seed),config=config)


def _fixed_normal_probe(model,sample,device):
    model.eval()
    with torch.no_grad():
        return normal_forward(model,to_device(sample['student'],device))['prediction'].cpu().clone()


def _calibrate(model,val_ds,indices,device,batch_size,workers):
    """Calibrate only scalar CF strength, never inspect CHECK for model selection."""
    if not indices:return {'enabled':False,'reason':'No disjoint calibration cohort','alpha':1.}
    loader=make_loader(Subset(val_ds,indices),batch_size,workers,False,401)
    errors={str(v):0. for v in (0.,.25,.5,1.,1.25)};denom=0
    model.eval()
    with torch.inference_mode():
        for b in loader:
            b=to_device(b,device);out=model(b['student']);valid=b['future_valid'].bool();y=b['y'].float()
            denom+=int(valid.sum())
            for a in (0.,.25,.5,1.,1.25):
                p=out['normal_prediction']+a*out['cf_increment']
                errors[str(a)]+=float(torch.where(valid,(p-y).abs(),torch.zeros_like(p)).sum())
    best=min(errors,key=lambda v:errors[v])
    return dict(enabled=True,alpha=float(best),calibration_positions=denom,
                calibration_errors={k:v/max(denom,1) for k,v in errors.items()},
                method='validation_calibration_only')


def _check_calibration(model,val_ds,indices,device,batch_size,workers,alpha):
    loader=make_loader(Subset(val_ds,indices),batch_size,workers,False,401)
    model.eval();n=0;err=0;normal=0;base=0
    with torch.inference_mode():
        for b in loader:
            b=to_device(b,device);out=model(b['student']);v=b['future_valid'].bool();y=b['y'].float()
            p=out['normal_prediction']+alpha*out['cf_increment']
            err+=float(torch.where(v,(p-y).abs(),torch.zeros_like(y)).sum())
            normal+=float(torch.where(v,(out['normal_prediction']-y).abs(),torch.zeros_like(y)).sum())
            base+=float(torch.where(v,(out['base']-y).abs(),torch.zeros_like(y)).sum());n+=int(v.sum())
    return dict(positions=n,mae_norm=err/max(n,1),normal_mae_norm=normal/max(n,1),
                base_mae_norm=base/max(n,1),cf_gain_norm=(normal-err)/max(n,1),alpha=alpha,
                check_used_for_selection=False)


def run_training(cache,teacher_root,output,config,model,seed=10,smoke=False,resume=None):
    set_seed(seed)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda' and not smoke:raise RuntimeError('Formal v9.3 training requires CUDA; --smoke permits CPU')
    model=model.to(device)
    root=Path(teacher_root)
    train=DirectViewDataset(cache,root,'train',limit=config['preparation']['limit'],
                            train_fraction=config['training']['train_fraction'])
    held=DirectViewDataset(cache,root,'train',limit=config['preparation']['limit'],
                           train_fraction=config['training']['train_fraction'],heldout=True)
    val=DirectViewDataset(cache,root,'val',limit=config['preparation']['val_limit'],views=1)
    selection=chronological_partitions(val.ds.window_starts[:len(val.ds)],
                                        config['model']['history'],config['model']['horizon'],smoke=smoke)
    val_sel=Subset(val,selection['selection'])
    if smoke:
        train=Subset(train,list(range(min(12,len(train)))))
        held=Subset(held,list(range(min(8,len(held)))))
        val_sel=Subset(val_sel,list(range(min(8,len(val_sel)))))
    bs=int(config['training']['batch_size'])
    workers=int(config['training']['workers'])
    meta=json.loads((root/'identity.json').read_text())
    sig=training_signature(root,config,model.mode,model.graph.mode,
             model.normal_query.net[0].in_features-10,model.graph.n,seed,meta)
    atomic_json(out/'run_identity.json',sig)
    stages=[(s,int(config['training']['epochs'][s])) for s in STAGES if int(config['training']['epochs'][s])>0]
    if model.mode=='single':stages=[item for item in stages if item[0]=='residual']
    if not stages or stages[0][0]!='residual':raise RuntimeError('Training must start from Residual')
    history=[];start_stage=0;start_epoch=0;last=None;best_by_stage={};normal_hash=None
    if resume:
        ck=torch.load(Path(resume),map_location='cpu',weights_only=False)
        if ck['signature']!=sig:raise RuntimeError('Resume configuration or Teacher identity differs')
        model.load_state_dict(ck['model'],strict=True)
        history=ck['history'];start_stage=ck['stage_index'];start_epoch=ck['next_epoch'];
        last=ck;best_by_stage=ck['best_by_stage'];normal_hash=ck.get('normal_hash')
        rng_restore(ck['rng'])
    for si,(stage,epochs) in enumerate(stages):
        if si<start_stage:continue
        names=stage_parameters(model,stage)
        if si>0 and not (last is not None and si==start_stage and start_epoch>0):
            p=out/f'best_{stages[si-1][0]}.pt'
            if not p.exists():raise FileNotFoundError('Prior stage best checkpoint missing: '+str(p))
            model.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['model'],strict=True)
        # normal parameters must be immutable from the first CF stage onward
        if si>0:
            if normal_hash is None:normal_hash=normal_fingerprint(model)
            if normal_fingerprint(model)!=normal_hash:raise RuntimeError('Normal fingerprint changed before stage')
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=float(config['training']['lr'])*(0.5 if si>=2 else 1.0),weight_decay=1e-4)
        scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and bool(config['training']['amp_fp16']))
        if last is not None and si==start_stage and start_epoch>0:
            optimizer.load_state_dict(last['optimizer']);scaler.load_state_dict(last['scaler'])
        if ((stage in ('cf_response','cf_joint') and bool(config['counterfactual']['paired_supervision']))
            or (stage in ('utility','cf_joint') and bool(config['counterfactual'].get('paired_utility',False)))):
            pair_data=DirectViewDataset(cache,root,'train',limit=config['preparation']['limit'],
                            pair=True,train_fraction=config['training']['train_fraction'])
            if smoke:pair_data=Subset(pair_data,list(range(min(8,len(pair_data)))))
            data=pair_data;paired=True
        else:data=train;paired=False
        bestscore=float(best_by_stage.get(stage,{}).get('score',float('inf')))
        start= start_epoch if si==start_stage else 0
        probe=next(iter(make_loader(val_sel,min(bs,4),0,False,seed)))
        anchor_before=_fixed_normal_probe(model,probe,device) if si>0 else None
        for ep in range(start,epochs):
            loader=make_loader(data,bs,workers,True,seed+ep+31*si)
            tr=_epoch(model,loader,stage,device,optimizer,scaler,config,True,paired=paired)
            if stage=='cf_response':
                ev=make_loader(held,min(bs,16),0,False,seed)
                stats=_epoch(model,ev,'cf_response',device,None,None,config,False,paired=False)
                # Selection is based on TRAIN-heldout CF response, never VAL-full.
                score=stats['loss']
            else:
                ev=make_loader(val_sel,min(bs,int(config['training']['eval_batch_size'])),0,False,seed)
                stats=_epoch(model,ev,stage,device,None,None,config,False,paired=False)
                score=stats['normal_mae_norm'] if stage=='residual' else stats['mae_norm']
            if si>0:
                if normal_hash!=normal_fingerprint(model):raise RuntimeError('Frozen Normal changed during CF training')
                anchor_after=_fixed_normal_probe(model,probe,device)
                torch.testing.assert_close(anchor_before,anchor_after,rtol=0,atol=0)
            rec=dict(stage=stage,epoch=ep+1,train=tr,selection=stats,selected_metric=score)
            history.append(rec)
            print('V93_EPOCH '+json.dumps(rec),flush=True)
            if score<bestscore:
                bestscore=score;best_by_stage[stage]={'score':bestscore,'epoch':ep+1}
                atomic_save(out/f'best_{stage}.pt',dict(model=model.state_dict(),stage=stage,
                      signature=sig,mode=model.mode,graph_mode=model.graph.mode,
                      version=VERSION,normal_hash=normal_hash))
            # Always save an exact continuation state after the optimizer step.
            nxt=ep+1
            stageidx=si if nxt<epochs else si+1
            if nxt>=epochs:nxt=0
            payload=dict(signature=sig,model=model.state_dict(),optimizer=optimizer.state_dict(),
                   scaler=scaler.state_dict(),history=history,stage_index=stageidx,next_epoch=nxt,
                   best_by_stage=best_by_stage,normal_hash=normal_hash,rng=rng_snapshot())
            atomic_save(out/'last.pt',payload)
        start_epoch=0;last=None
    model.load_state_dict(torch.load(out/f'best_{stages[-1][0]}.pt',map_location='cpu',weights_only=False)['model'])
    model.eval()
    calibration=_calibrate(model,val,selection['calibration'],device,min(bs,16),0)
    atomic_json(out/'calibration.json',calibration)
    # CHECK is reported once, never used to choose alpha, model stage or checkpoint.
    check=_check_calibration(model,val,selection['check'],device,min(bs,16),0,calibration['alpha'])
    atomic_json(out/'check_report.json',check)
    atomic_json(out/'history.json',history)
    summary=dict(version=VERSION,mode=model.mode,graph_mode=model.graph.mode,
         stages=best_by_stage,calibration=calibration,check=check,smoke=smoke,
         note='Synthetic smoke results are not formal forecasting outcomes')
    atomic_json(out/'summary.json',summary)
    print('V93_TRAIN_PASS '+json.dumps(summary),flush=True)
    return summary
