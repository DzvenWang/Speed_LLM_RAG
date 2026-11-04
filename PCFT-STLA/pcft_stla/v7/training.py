"""v7 staged trainer, train-only utility split, no test-based selection, durable resume."""
from __future__ import annotations
import copy,datetime,json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from .utility import supervised_utility_loss
from pcft_stla.v6.training import to_device,seed_all
from pcft_stla.v6.config import json_write

STAGES=('local','propagation','utility','joint')

def time_partition(ds,fraction=.8,embargo=24):
    meta=ds.meta
    if 'window_start' not in meta:raise ValueError('Required immutable TRAIN window_start metadata missing')
    starts=np.asarray(meta.window_start,dtype=np.int64)
    unique=np.unique(starts)
    if len(unique)<10:raise ValueError('Not enough distinct TRAIN timestamps for utility holdout')
    boundary=unique[int(len(unique)*fraction)]
    # Each window includes the 12 history and 12 future steps.
    first=np.flatnonzero(starts+int(embargo)<boundary)
    second=np.flatnonzero(starts>=boundary)
    if len(first)<2 or len(second)<2:raise ValueError('Insufficient utility partition after temporal embargo')
    # Strict nonoverlap of complete windows including their target horizons.
    if int(starts[first].max())+24>int(starts[second].min()):raise AssertionError('Candidate and utility windows overlap')
    return first.tolist(),second.tolist(),{'train_n':len(first),'utility_n':len(second),
        'train_last_window':int(starts[first].max()),'utility_first_window':int(starts[second].min()),'embargo':int(embargo)}


def trainable_for(model,stage,cfg):
    model.set_stage(stage)
    for p in model.parameters():p.requires_grad_(False)
    model.anchor.set_trainable_tail(0)
    if stage=='local':modules=[model.support,model.local_head]
    elif stage=='propagation':modules=[model.support,model.local_head,model.propagation,model.prop_head]
    elif stage=='utility':modules=[model.controller]
    elif stage=='joint':modules=[model.support,model.local_head,model.propagation,model.prop_head]
    else:raise ValueError(stage)
    for m in modules:
        for p in m.parameters():p.requires_grad_(True)
    if stage=='joint' and cfg['training'].get('joint_unfreeze_stla_layers',0):
        model.anchor.set_trainable_tail(int(cfg['training']['joint_unfreeze_stla_layers']))
    names=[name for name,p in model.named_parameters() if p.requires_grad]
    if not names:raise RuntimeError('No trainable parameters')
    return names


def optimizer_for(model,cfg,stage):
    bb=[];new=[]
    for name,p in model.named_parameters():
        if p.requires_grad:(bb if name.startswith('anchor.base.') else new).append(p)
    groups=[]
    if bb:groups.append({'params':bb,'lr':float(cfg['training']['lr_backbone'])})
    if new:groups.append({'params':new,'lr':float(cfg['training']['lr_controller'] if stage=='utility' else cfg['training']['lr_new'] * (.3 if stage=='joint' else 1))})
    return torch.optim.AdamW(groups,weight_decay=float(cfg['training']['weight_decay']))


def scaler_for(device,cfg):
    return torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and bool(cfg['training']['amp']))


def autocast_for(device,cfg):
    from contextlib import nullcontext
    return torch.autocast('cuda',dtype=torch.float16) if device.type=='cuda' and bool(cfg['training']['amp']) else nullcontext()


def prediction_loss(output,batch,cfg,stage):
    valid=batch['future_valid'].bool()&torch.isfinite(batch['y_norm'])
    target=torch.where(valid,batch['y_norm'],0.)
    err=torch.where(valid,output['prediction'].float()-target.float(),0.)
    abs_err=err.abs()
    n=valid.sum().clamp_min(1)
    mae=abs_err.sum()/n
    dm=batch['direct_mask'].bool()
    if dm.ndim==2:dm=dm[:,None,:].expand_as(valid)
    if dm.shape!=valid.shape:raise ValueError('direct_mask shape mismatch')
    mv=valid&dm
    masked=(abs_err*mv).sum()/mv.sum().clamp_min(1)
    horizon_weights=torch.linspace(1.,1.5,abs_err.shape[1],device=abs_err.device)[None,:,None]
    horizon=(abs_err*horizon_weights).sum()/(valid.float()*horizon_weights).sum().clamp_min(1)
    lc=cfg['training']['loss']
    loss=mae+float(lc['masked'])*masked+float(lc['horizon'])*horizon
    if stage in ('local','propagation','joint'):
        reg=(output['local_delta'].square()+output['prop_delta'].square()).mean()
        loss=loss+float(lc['correction_regularization'])*reg
    if stage=='utility':
        loss=loss+supervised_utility_loss(output,batch,float(lc['gain']),float(lc['risk']),float(lc['action']))
    return loss


class RunningMetrics:
    def __init__(self,device):
        self.acc=torch.zeros(14,device=device,dtype=torch.float64)
    def update(self,output,batch,mean,std):
        pred=output['prediction'].float()*std+mean
        y=batch['y_raw'].float()
        v=batch['future_valid'].bool()&torch.isfinite(y)
        d=batch['direct_mask'].bool()
        if d.ndim==2:d=d[:,None,:].expand_as(v)
        if d.shape!=v.shape:raise ValueError('Invalid direct_mask')
        error=torch.where(v,pred-y,0.)
        a=error.abs();sel=v&d
        q=self.acc
        q[0]+=(a).sum();q[1]+=(error.square()).sum();q[2]+=v.sum()
        q[3]+=(a*sel).sum();q[4]+=sel.sum()
        q[5]+=a[:,-1].sum();q[6]+=v[:,-1].sum()
        # All action metrics include ALL valid positions, no post-selection restriction.
        can=output['candidates'].float()*std+mean
        ca=torch.where(v[...,None],(can-torch.where(v,y,0.)[...,None]).abs(),0.)
        for i in range(3):q[7+i]+=(ca[...,i]*v).sum()
        # Harmful corrections relative to baseline and local-only candidate.
        q[10]+=((a>ca[...,0]+1e-6)&v).sum()
        q[11]+=((ca[...,2]>ca[...,1]+1e-6)&v).sum()
        q[12]+=(output['policy']['weights'][...,2]*v).sum()
        q[13]+=(output['policy']['weights'][...,0]*v).sum()
    def finish(self):
        a=self.acc.cpu().tolist()
        if a[2]<=0:raise RuntimeError('No valid evaluation targets')
        div=lambda x,y:float(x/y) if y else None
        return {'mae':div(a[0],a[2]),'rmse':math.sqrt(a[1]/a[2]),'masked_mae':div(a[3],a[4]),
                'h12_mae':div(a[5],a[6]),'base_mae':div(a[7],a[2]),'local_mae':div(a[8],a[2]),
                'propagation_mae':div(a[9],a[2]),'harmful_vs_base':div(a[10],a[2]),
                'propagation_harmful_vs_local':div(a[11],a[2]),
                'propagation_weight':div(a[12],a[2]),'no_correction_weight':div(a[13],a[2]),
                'valid_count':int(a[2]),'masked_count':int(a[4])}


def loader(ds,batch_size,shuffle,workers,device):
    return DataLoader(ds,batch_size=int(batch_size),shuffle=shuffle,num_workers=int(workers),
                      pin_memory=device.type=='cuda',persistent_workers=workers>0)


def run_epoch(model,ds,stage,cfg,device,mean,std,opt=None,scaler=None,max_batches=None):
    train=opt is not None
    model.train(train)
    if not train:model.eval()
    metric=RunningMetrics(device);loss_acc=torch.zeros(2,device=device)
    batch_size=cfg['training']['batch_size'] if train else cfg['runtime']['eval_batch']
    ld=loader(ds,batch_size,train and stage!='utility',cfg['runtime']['workers'] if train else 0,device)
    for i,batch in enumerate(ld):
        if max_batches is not None and i>=max_batches:break
        batch=to_device(batch,device)
        if train:opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with autocast_for(device,cfg):
                output=model(batch)
                loss=prediction_loss(output,batch,cfg,stage)
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss {stage}, batch {i}')
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],float(cfg['training']['grad_clip']))
                scaler.step(opt);scaler.update()
        with torch.no_grad():
            metric.update(output,batch,mean,std)
            loss_acc[0]+=loss.detach()
            loss_acc[1]+=1
        if train and (i+1)%int(cfg['runtime']['print_every'])==0:
            print(f'V7_PROGRESS stage={stage} batches={i+1} local_time={datetime.datetime.now().isoformat()}',flush=True)
    ret=metric.finish()
    vals=loss_acc.cpu().tolist()
    ret['loss']=float(vals[0]/max(vals[1],1.))
    return ret


def capture_rng():
    return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy':np.random.get_state(),'python':random.getstate()}


def restore_rng(r):
    torch.set_rng_state(r['torch']);np.random.set_state(r['numpy']);random.setstate(r['python'])
    if r['cuda'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(r['cuda'])


def atomic_save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial.pt')
    torch.save(obj,tmp);os.replace(tmp,path)


def train_pipeline(model,datasets,cfg,identity,out_dir,device,*,resume=None,smoke=False):
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    tr_idx,utility_idx,partition=time_partition(datasets['train'],cfg['training']['candidate_train_fraction'],cfg['training']['temporal_embargo_steps'])
    json_write(out_dir/'train_partition.json',partition)
    train_ds=Subset(datasets['train'],tr_idx);utility_ds=Subset(datasets['train'],utility_idx)
    val_ds=datasets['val']
    std=float(identity['std']);mean=float(identity['mean'])
    previous= None
    if resume:
        previous=torch.load(resume,map_location='cpu',weights_only=False)
        if previous.get('schema')!='pcft_v7_resume_v1' or previous.get('identity')!=identity or previous.get('config')!=cfg:
            raise RuntimeError('Resume checkpoint identity/configuration mismatch')
        model.load_state_dict(previous['model'],strict=True)
        restore_rng(previous['rng'])
    histories=[]
    names=list(STAGES)
    from_index=names.index(previous['stage']) if previous else 0
    for idx,stage in enumerate(names):
        if idx<from_index:continue
        print('V7_STAGE_START',stage,flush=True)
        trainable_for(model,stage,cfg)
        opt=optimizer_for(model,cfg,stage);scaler=scaler_for(device,cfg)
        epoch_start=1;best=float('inf');best_epoch=0
        if previous and stage==previous['stage'] and previous['optimizer'] is not None:
            opt.load_state_dict(previous['optimizer']);scaler.load_state_dict(previous['scaler'])
            epoch_start=int(previous['epoch'])+1
            best=float(previous['best']);best_epoch=int(previous['best_epoch'])
        max_epochs=int(cfg['training']['stages'][stage]['max_epochs'])
        patience=int(cfg['training']['stages'][stage]['patience'])
        if smoke:max_epochs=min(max_epochs,1)
        for epoch in range(epoch_start,max_epochs+1):
            t0=time.perf_counter()
            # Stage C learns only on disjoint TRAIN utility block; candidate modules are frozen.
            active=utility_ds if stage=='utility' else train_ds
            train=run_epoch(model,active,stage,cfg,device,mean,std,opt,scaler,max_batches=1 if smoke else None)
            # Stage D alternates disjoint-block controller updates, never uses candidate-fitting labels for controller updates.
            if stage=='joint' and cfg['training'].get('joint_refresh_utility',True):
                trainable_for(model,'utility',cfg)
                uopt=optimizer_for(model,cfg,'utility');uscaler=scaler_for(device,cfg)
                _=run_epoch(model,utility_ds,'utility',cfg,device,mean,std,uopt,uscaler,max_batches=1 if smoke else None)
                trainable_for(model,'joint',cfg)
            val=run_epoch(model,val_ds,'joint' if stage in ('utility','joint') else stage,cfg,device,mean,std,
                          max_batches=1 if smoke else None)
            row={'stage':stage,'epoch':epoch,'elapsed_s':round(time.perf_counter()-t0,2),'train':train,'val':val}
            with (out_dir/'epoch_history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print('V7_EPOCH',json.dumps(row),flush=True)
            if val['mae']+float(cfg['training']['min_delta'])<best:
                best=val['mae'];best_epoch=epoch
                atomic_save(out_dir/f'{stage}_best.pt',{'schema':'pcft_v7_best_v1','stage':stage,'model':model.state_dict(),
                         'identity':identity,'config':cfg,'val':val,'epoch':epoch})
            state={'schema':'pcft_v7_resume_v1','stage':stage,'epoch':epoch,'model':model.state_dict(),
                   'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'best':best,'best_epoch':best_epoch,
                   'rng':capture_rng(),'identity':identity,'config':cfg}
            atomic_save(out_dir/'last.pt',state)
            if epoch-best_epoch>=patience:break
        best_file=out_dir/f'{stage}_best.pt'
        if not best_file.exists():raise RuntimeError(f'No best for stage {stage}')
        ck=torch.load(best_file,map_location=device,weights_only=False)
        model.load_state_dict(ck['model'],strict=True)
        print('V7_STAGE_COMPLETE',stage,'best_epoch',ck['epoch'],'val',ck['val']['mae'],flush=True)
        if idx+1<len(names):
            # A durable stage transition. If interruption happens, no stage is run twice.
            nxt={'schema':'pcft_v7_resume_v1','stage':names[idx+1],'epoch':0,'model':model.state_dict(),
                 'optimizer':None,'scaler':None,'best':float('inf'),'best_epoch':0,'rng':capture_rng(),
                 'identity':identity,'config':cfg}
            atomic_save(out_dir/'last.pt',nxt)
        previous=None
    # Strict test isolation: only run after ALL stages and validation selection are complete.
    model.load_state_dict(torch.load(out_dir/'joint_best.pt',map_location=device,weights_only=False)['model'],strict=True)
    model.set_stage('joint');model.eval()
    results=[]
    for severity in cfg['experiment']['severities']:
        met=run_epoch(model,datasets[f'test_s{int(severity):02d}'],'joint',cfg,device,mean,std,
                      max_batches=1 if smoke else None)
        row={'severity':int(severity),'target':cfg['experiment']['target'],'selection':'joint_best_on_validation',**met}
        results.append(row)
        json_write(out_dir/'test_results.json',results)
        print('V7_TEST',json.dumps(row),flush=True)
    print('V7_COMPLETE',str(out_dir),flush=True)
    return results
