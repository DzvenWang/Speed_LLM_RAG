"""CoRSTA training, selection, locked evaluation and resume.

Never reads query futures in model forward; only uses targets for TRAIN loss
and metric evaluation. Teacher memory stays immutable.
"""
from __future__ import annotations
import copy,gc,json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .batching import select_batch,smaller_size,is_oom
from .deep import CoRSTA
from .student import SplitSTLA
from .stla import load_pretrained_stla
from .v7.runner import (graph_and_memory,load_network,loader,as_device,atomic_save,atomic_json,
                         selection_indices,train_cache,seed_everything)
from .v7.replay import ReplayDataset,prepare_replay
from .v7.sampling import replay_train_loader
from .v7.losses import cfrm_auxiliary
from .data import SupportSplit
from .config import cache_path,file_sha256

VERSION='0.1.0'


def build_model(legacy_args,cfg,device):
    _,graph,bc=graph_and_memory(legacy_args,device)
    source,stla_bc=load_pretrained_stla(Path(legacy_args.root),Path(legacy_args.backbone_checkpoint),device)
    if int(source.num_nodes)!=int(legacy_args.nodes):raise ValueError('STLA nodes mismatch')
    tail=int(cfg['model']['stla_tail_layers'])
    student=SplitSTLA(source,tail).to(device)
    s0=torch.load(legacy_args.run/'s0.pt',map_location='cpu',weights_only=False)
    if s0.get('backbone_sha256')!=file_sha256(legacy_args.backbone_checkpoint):
        raise RuntimeError('Bootstrap S0 identity invalid')
    student.mask_proj.load_state_dict(s0['mask_proj'])
    init=str(cfg['model']['legacy_c1_init'])
    if init!='best_c1':raise ValueError('Supported C1 initialization: best_c1')
    legacy=load_network(legacy_args,device,graph,stla_bc,previous='c1')
    model=CoRSTA(student,legacy,graph,
         token_dim=int(cfg['model']['token_dim']),graph_dim=int(cfg['model']['graph_dim']),
         heads=int(cfg['model']['bridge_heads'])).to(device)
    return model,stla_bc


def configure_model(model,phase,cfg):
    if phase=='warmup':
        model.configure_student('frozen')
        for p in model.student.parameters():p.requires_grad_(False)
    elif phase=='joint':
        model.configure_student(str(cfg['model']['student_train_phase']),int(cfg['model']['stla_tail_layers']))
    else:raise ValueError(phase)
    model.c1.freeze_stage('c1')
    return [p for p in model.parameters() if p.requires_grad]


def build_optimizer(model,phase,cfg):
    lr=cfg['training'];groups=[]
    entries=[('student.base.output_proj',lr['lr_stla_head']),
             ('student.base.attn_layers_st',lr['lr_stla_tail']),
             ('student.base',lr['lr_stla_tail']),
             ('student.mask_proj',lr['lr_mask_proj']),
             ('c1',lr['lr_c1']),('support',lr['lr_c1']),
             ('c2',lr['lr_c2']),('bridge1',lr['lr_bridge']),('bridge2',lr['lr_bridge']),
             ('fusion',lr['lr_bridge'])]
    used=set()
    for prefix,rate in entries:
        ps=[p for name,p in model.named_parameters() if p.requires_grad and
            (name==prefix or name.startswith(prefix+'.'))]
        # STLA prefix blocks are frozen when using partial tuning.
        ps=[p for p in ps if id(p) not in used]
        if ps:groups.append({'params':ps,'lr':float(rate)});used.update(map(id,ps))
    missing=[name for name,p in model.named_parameters() if p.requires_grad and id(p) not in used]
    if missing:raise RuntimeError(f'Optimizer does not own parameters: {missing[:10]}')
    return torch.optim.AdamW(groups,weight_decay=float(lr['weight_decay']))


def read_dataset(args,split,severity=None):
    src=cache_path(args.config,'evaluation',severity) if severity is not None else cache_path(args.config,'formal')
    real='test' if severity is not None else split
    ds=SupportSplit(src,real,limit=args.limit('test' if severity is not None else split))
    records={k:np.load(args.run/'teacher_cache'/real.replace('test',f'test_s{severity}' if severity is not None else 'test')/
                 f'{real}_{k}.npy',mmap_mode='r') for k in ('base','latent')}
    replay=args.run/'paired_replay'/(f'test_s{severity}' if severity is not None else real)
    return ReplayDataset(ds,records,replay,args.replay_chunk),ds


def make_data(args,device,split,severity=None):
    if severity is not None:
        train_cache(args,device,'test',severity)
        prepare_replay(args,device,'test',severity)
    dataset,raw=read_dataset(args,split,severity)
    return dataset,raw


def get_rnd():
    return {'torch':torch.get_rng_state(),
       'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
       'numpy':np.random.get_state(),'python':random.getstate()}


def set_rnd(s):
    torch.set_rng_state(s['torch'])
    if s['cuda'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(s['cuda'])
    np.random.set_state(s['numpy']);random.setstate(s['python'])


def train_loss(out,b,bc,cfg):
    std=float(bc['std']);mean=float(bc['mean'])
    valid=b['future_valid'].float()
    target=b['y_raw'].float()
    pred=out['refined'].float()*std+mean
    err=pred-target
    denom=valid.sum().clamp_min(1)
    mae=(err.abs()*valid).sum()/denom
    rmse=torch.sqrt((err.square()*valid).sum()/denom+1e-8)
    main=mae+.05*rmse
    p0=(out['backbone'].float()*std+mean)
    base=((p0-target).abs()*valid).sum()/denom
    # Frozen teacher output is only a regularization reference, not the target.
    anchor=(((out['backbone']-b['cached_base']).float()*std).abs()*valid).sum()/denom
    aux=cfrm_auxiliary(out['legacy'],b,mean,std,wrong_pair=('cf_wrong_increment' in out['legacy']))
    tr=cfg['training']
    total=main+float(tr['base_aux_weight'])*base+float(tr['anchor_weight'])*anchor+float(tr['c1_aux_weight'])*aux
    return total,{'mae':mae.detach(),'rmse':rmse.detach(),'base':base.detach(),'anchor':anchor.detach(),'c1_aux':aux.detach()}


@torch.no_grad()
def evaluate(model,dataset,device,bc,batch_size,indices=None,workers=0):
    model.eval();stats={k:[0.]*7 for k in ('backbone','c1_posthoc','refined')}
    dl=loader(dataset,batch_size,workers,subset=indices)
    for raw in dl:
        b=as_device(raw,device)
        out=model(b)
        for k in stats:
            pred=out[k].float()*float(bc['std'])+float(bc['mean'])
            target=b['y_raw'].float();valid=b['future_valid'].bool()
            err=pred-target
            s=stats[k]
            s[0]+=float((err.abs()*valid).sum());s[1]+=float((err.square()*valid).sum());s[2]+=int(valid.sum())
            direct=b['direct_mask'].bool()
            if direct.ndim==2:direct=direct[:,None]
            if direct.shape not in ((len(valid),1,valid.shape[-1]),valid.shape):
                raise ValueError(f'direct_mask incorrect shape: {direct.shape}')
            dm=direct&valid
            s[3]+=float((err.abs()*dm).sum());s[4]+=int(dm.sum())
            s[5]+=float((err[:,-1].abs()*valid[:,-1]).sum());s[6]+=int(valid[:,-1].sum())
    return {k:{'mae':s[0]/max(1,s[2]),'rmse':math.sqrt(s[1]/max(1,s[2])),
                'masked_mae':s[3]/max(1,s[4]),'h12_mae':s[5]/max(1,s[6])} for k,s in stats.items()}


def save_checkpoint(path,model,optimizer,scaler,phase,epoch,best,best_epoch,wait,batch_size,cfg_hash,completed=True):
    ck={'version':VERSION,'phase':phase,'epoch':epoch,'model':model.state_dict(),
       'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
       'best':best,'best_epoch':best_epoch,'wait':wait,'batch_size':batch_size,
       'config_sha256':cfg_hash,'random_state':get_rnd(),'completed_epoch':completed}
    atomic_save(ck,path)


def train_phase(args,cfg,device,outdir,phase,cfg_hash):
    data_train,_=make_data(args,device,'train')
    data_val,raw_val=make_data(args,device,'val')
    selection,locked=selection_indices(raw_val)
    model,bc=build_model(args,cfg,device)
    if phase=='joint':
        source=outdir/'best_warmup.pt'
        if not source.is_file():raise RuntimeError('Joint requires completed warmup')
        ck=torch.load(source,map_location='cpu',weights_only=False)
        if ck['config_sha256']!=cfg_hash:raise RuntimeError('Warmup identity mismatch')
        model.load_state_dict(ck['model'],strict=True)
    trainable=configure_model(model,phase,cfg)
    optimizer=build_optimizer(model,phase,cfg)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and bool(cfg['training']['amp']))
    last=outdir/f'last_{phase}.pt';best_path=outdir/f'best_{phase}.pt'
    epochs=int(cfg['training'][phase+'_epochs']);patience=int(cfg['training']['patience'])
    start=1;wait=0;best=float('inf');best_epoch=0
    selected_batch=None
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=False)
        if ck['config_sha256']!=cfg_hash or ck['phase']!=phase:raise RuntimeError('Checkpoint identity mismatch')
        model.load_state_dict(ck['model'],strict=True)
        optimizer.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
        best=float(ck['best']);best_epoch=int(ck['best_epoch']);wait=int(ck['wait'])
        start=int(ck['epoch'])+1;selected_batch=int(ck['batch_size']);set_rnd(ck['random_state'])
        print(f'CORSTA_RESUME phase={phase} epoch={start} best_epoch={best_epoch} batch={selected_batch}',flush=True)
    else:
        # Save original no-training starting state and its Validation Selection metric.
        result=evaluate(model,data_val,device,bc,int(cfg['training']['eval_batch_size']),selection)
        best=result['refined']['mae'];best_epoch=0
        atomic_save({'version':VERSION,'phase':phase,'epoch':0,'model':model.state_dict(),
                     'config_sha256':cfg_hash,'best_selection_mae':best},best_path)
        print(f'CORSTA_BASELINE phase={phase} selection='+json.dumps(result),flush=True)
    if selected_batch is None:
        mode=cfg['training']['batch_mode']
        if mode=='auto':
            selected_batch,attempts=select_batch(model,data_train,device,
                bool(cfg['training']['amp']),int(cfg['training']['batch_initial']),
                float(cfg['training']['batch_capacity_fraction']))
        elif mode=='fixed':
            selected_batch=int(cfg['training']['batch_initial'])
            if selected_batch not in (16,32,64,128):raise ValueError('Batch must be one of 16/32/64/128')
            attempts=[]
        else:raise ValueError('batch_mode must be auto or fixed')
        atomic_json({'selected_batch':selected_batch,'attempts':attempts,'phase':phase},outdir/f'batch_probe_{phase}.json')
    effective=max(selected_batch,int(cfg['training']['effective_batch']))
    accum=math.ceil(effective/selected_batch)
    if not last.exists():
        save_checkpoint(last,model,optimizer,scaler,phase,0,best,best_epoch,wait,selected_batch,cfg_hash)
    for ep in range(start,epochs+1):
        while True:
            tic=time.perf_counter();valid_count=0;err_sum=0.;total_aux=0.;n_batch=0
            try:
                model.train()
                if phase=='warmup':model.student.eval()
                dl=replay_train_loader(data_train,selected_batch,int(cfg['training']['worker_count']))
                optimizer.zero_grad(set_to_none=True)
                if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
                for i,raw in enumerate(dl,1):
                    b=as_device(raw,device)
                    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=scaler.is_enabled()):
                        output=model(b)
                        loss,detail=train_loss(output,b,bc,cfg)
                    if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite CoRSTA loss epoch={ep} batch={i}')
                    # Last partial accumulation scaled by effective batch estimate.
                    actual_steps=min(accum,len(dl)-((i-1)//accum)*accum)
                    scaled=loss/max(1,actual_steps)
                    if scaler.is_enabled():scaler.scale(scaled).backward()
                    else:scaled.backward()
                    if i%accum==0 or i==len(dl):
                        if scaler.is_enabled():scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(trainable,float(cfg['training']['max_grad_norm']))
                        if scaler.is_enabled():scaler.step(optimizer);scaler.update()
                        else:optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    n_valid=int(b['future_valid'].sum());valid_count+=n_valid
                    err_sum+=float(detail['mae'])*n_valid;total_aux+=float(detail['c1_aux'])*n_valid
                    n_batch+=1
                del dl
                break
            except (RuntimeError,MemoryError) as exc:
                if device.type!='cuda' or not is_oom(exc) or cfg['training']['batch_mode']!='auto':raise
                next_batch=smaller_size(selected_batch)
                if next_batch is None:raise RuntimeError('CUDA OOM at batch 16: no safe lower configured size') from exc
                print(f'CORSTA_RUNTIME_OOM phase={phase} epoch={ep} batch={selected_batch} fallback={next_batch} error={str(exc)[:160]}',flush=True)
                # Roll back *all* model, optimizer, scaler and random state to
                # last completed epoch, so partial failed epoch never contaminates training.
                optimizer.zero_grad(set_to_none=True)
                if 'dl' in locals():del dl
                if 'b' in locals():del b
                if 'output' in locals():del output
                if 'loss' in locals():del loss
                gc.collect();torch.cuda.empty_cache()
                ck=torch.load(last,map_location='cpu',weights_only=False)
                model.load_state_dict(ck['model'],strict=True)
                optimizer.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
                set_rnd(ck['random_state'])
                selected_batch=next_batch;accum=math.ceil(effective/selected_batch)
                atomic_json({'selected_batch':selected_batch,'runtime_fallback':True,'epoch':ep},outdir/f'batch_probe_{phase}.json')
        result=evaluate(model,data_val,device,bc,int(cfg['training']['eval_batch_size']),selection)
        metric=result['refined']['mae']
        if metric<best-1e-7:
            best=metric;best_epoch=ep;wait=0
            atomic_save({'version':VERSION,'phase':phase,'epoch':ep,'model':model.state_dict(),
                         'config_sha256':cfg_hash,'best_selection_mae':best},best_path)
        else:wait+=1
        peak=(torch.cuda.max_memory_allocated(device)/1024**3 if device.type=='cuda' else 0)
        print(f'CORSTA_EPOCH phase={phase} epoch={ep}/{epochs} batch={selected_batch} accum={accum} '
              f'train_mae_raw={err_sum/max(valid_count,1):.6f} c1_aux={total_aux/max(valid_count,1):.6f} '
              f'selection_mae={metric:.6f} selection_backbone={result["backbone"]["mae"]:.6f} '
              f'selection_c1={result["c1_posthoc"]["mae"]:.6f} best={best:.6f} '
              f'throughput={len(data_train)/max(time.perf_counter()-tic,1):.2f}q/s '
              f'gpu_peak_gib={peak:.3f} wait={wait}',flush=True)
        save_checkpoint(last,model,optimizer,scaler,phase,ep,best,best_epoch,wait,selected_batch,cfg_hash)
        if wait>=patience:break
    atomic_json({'phase':phase,'best_selection_mae':best,'best_epoch':best_epoch},outdir/f'selection_{phase}.json')
    print(f'CORSTA_STAGE_PASS phase={phase} best_epoch={best_epoch} best_selection_mae={best:.6f}',flush=True)


def load_final(args,cfg,device,outdir,cfg_hash):
    model,bc=build_model(args,cfg,device)
    ck=torch.load(outdir/'best_joint.pt',map_location='cpu',weights_only=False)
    if ck['config_sha256']!=cfg_hash:raise RuntimeError('Final model identity mismatch')
    model.load_state_dict(ck['model'],strict=True)
    model.eval()
    return model,bc,ck


def lock(args,cfg,device,outdir,cfg_hash):
    if (outdir/'model_lock.json').exists():raise RuntimeError('Lock already exists; cannot change model or repeat locked check')
    model,bc,ck=load_final(args,cfg,device,outdir,cfg_hash)
    data,raw=make_data(args,device,'val')
    _,indices=selection_indices(raw)
    metrics=evaluate(model,data,device,bc,int(cfg['training']['eval_batch_size']),indices)
    atomic_json({'version':VERSION,'model':'best_joint.pt','model_sha256':file_sha256(outdir/'best_joint.pt'),
                 'config_sha256':cfg_hash,'selected_mode':'refined','locked_metrics':metrics},outdir/'model_lock.json')
    print('CORSTA_LOCKED_CHECK '+json.dumps(metrics),flush=True)


def test_severity(args,cfg,device,outdir,cfg_hash,severity):
    lockfile=outdir/'model_lock.json'
    if not lockfile.exists():raise RuntimeError('TEST forbidden before frozen model lock')
    identity=json.loads(lockfile.read_text())
    if identity['config_sha256']!=cfg_hash or identity['model_sha256']!=file_sha256(outdir/'best_joint.pt'):
        raise RuntimeError('Model changed after lock')
    model,bc,_=load_final(args,cfg,device,outdir,cfg_hash)
    data,_=make_data(args,device,'test',severity)
    metrics=evaluate(model,data,device,bc,int(cfg['training']['eval_batch_size']))
    atomic_json({'version':VERSION,'severity':severity,'metrics':metrics,'selected':metrics['refined']},
                outdir/'test'/f's{severity}.json')
    print(f'CORSTA_TEST_PASS severity=s{severity} '+json.dumps(metrics),flush=True)


def report(cfg,outdir):
    test={}
    for severity in cfg['experiment']['severities']:
        p=outdir/'test'/f's{severity}.json'
        if not p.exists():raise RuntimeError(f'Missing TEST severity: {severity}')
        test[f's{severity}']=json.loads(p.read_text())['metrics']
    summary={m:{k:sum(test[s][m][k] for s in test)/len(test) for k in ('mae','rmse','masked_mae','h12_mae')}
             for m in ('backbone','c1_posthoc','refined')}
    atomic_json({'version':VERSION,'test':test,'mean_severities':summary},outdir/'summary.json')
    lines=['# CoRSTA evaluation','', '| Model | MAE | RMSE | Masked MAE | H12 MAE |','|---|---:|---:|---:|---:|']
    for name,v in summary.items():lines.append(f'| {name} | {v["mae"]:.4f} | {v["rmse"]:.4f} | {v["masked_mae"]:.4f} | {v["h12_mae"]:.4f} |')
    (outdir/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('CORSTA_REPORT_PASS '+json.dumps(summary),flush=True)
