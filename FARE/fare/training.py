"""Single full-FARE staged optimization with AMP, efficient snapshot replay and resume."""
from pathlib import Path
import random,time,json,math
import numpy as np,torch
import torch.nn.functional as F
from .protocol import make_loader,device_batch
from .utility_teacher import generate_utility_teacher,replay_utility_loss
from .support_teacher import build_support_teacher,support_auxiliary_loss
from .config import dump_json,hash_file


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def forecasting_loss(out,batch,cfg):
    prediction=out['prediction'];target=batch['target'];valid=batch['future_valid'].bool()
    diff=(prediction.float()-target.float()).abs()
    den=valid.sum().clamp_min(1)
    mae=(diff*valid).sum()/den
    direct=valid&batch['direct_mask'][:,None,:].bool()
    masked=(diff*direct).sum()/direct.sum().clamp_min(1)
    tail=F.smooth_l1_loss(prediction.float(),target.float(),reduction='none')
    tail=(tail*valid).sum()/den
    loss=mae+cfg['training'].get('masked_weight',.2)*masked+cfg['training'].get('tail_weight',.02)*tail
    # Lightweight propagation support regularizer; ground coverage, not physical wave labels.
    demand=out['support']['demand']
    coverage=out['support']['forward_coverage']
    support_reg=(demand-(1-coverage).detach()).abs().mean()
    loss=loss+cfg['training'].get('support_weight',.02)*support_reg
    return loss,{'mae':mae.detach(),'masked':masked.detach(),
                 'tail':tail.detach(),'support':support_reg.detach()}


def checkpoint(path,model,opt,cfg,state,metadata,scaler=None):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'version':'6.0.0','state_dict':model.state_dict(),
                'optimizer_state':opt.state_dict(),'scaler_state':scaler.state_dict() if scaler is not None else None,
                'config':cfg,'training_state':state,'metadata':metadata,
                'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
                'python_rng':random.getstate(),
                'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},p.with_suffix(p.suffix+'.tmp'))
    p.with_suffix(p.suffix+'.tmp').replace(p)


def _set_stage(model,stage):
    """Stage-specific trainable parameters; never unfreeze official STLAformer."""
    for name,param in model.named_parameters():
        if name.startswith('backbone.base.'):
            param.requires_grad_(False)
        elif stage=='support':
            param.requires_grad_(name.startswith(('base_proj.','support.',
                                                 'support_delta.','vulnerability.','deficit.')))
        elif stage in ('retrieval','utility'):
            param.requires_grad_(name.startswith(('ranker.','selector.','retriever.',
                                                 'deficit.','support.','vulnerability.')))
        elif stage=='risk':
            param.requires_grad_(name.startswith(('risk.','selector.null.')))
        else:
            param.requires_grad_(True)

def fit(model,memory,train_ds,val_ds,device,cfg,run_dir,metadata,resume=None):
    run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=True)
    tc=cfg['training'];stage_epochs=tc['stage_epochs'];stages=list(stage_epochs)
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=float(tc['lr']),
                          weight_decay=float(tc['weight_decay']))
    amp=bool(tc.get('amp',True)) and torch.device(device).type=='cuda'
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    best=float('inf');steps=0;start_stage=0;start_epoch=0
    if tc.get('initialize_from'):
        raise ValueError('FARE warm-start forbidden: use newly initialized FARE parameters')
    if resume is None:
        print('FARE6_INITIALIZATION '+json.dumps({
            'fare_parameters':'fresh_random_seeded',
            'backbone':'frozen_pretrained_STLAformer',
            'historical_evidence':'prebuilt_chronological_OOF',
            'prior_fare_checkpoint_loaded':False,
        }), flush=True)
    if resume:
        saved=torch.load(resume,map_location=device,weights_only=False)
        if saved.get('version')!='6.0.0':raise RuntimeError('FARE v6 only resumes v6 checkpoints; use initialize_from for v5 weights')
        if saved['metadata'].get('backbone_sha256')!=metadata.get('backbone_sha256'):
            raise RuntimeError('Checkpoint Backbone hash mismatch')
        model.load_state_dict(saved['state_dict'],strict=True);opt.load_state_dict(saved['optimizer_state'])
        if saved.get('scaler_state'):scaler.load_state_dict(saved['scaler_state'])
        state=saved['training_state'];best=state['best_val'];steps=state.get('steps',0)
        start_stage=state['stage_idx'];start_epoch=state['stage_epoch']+1
        if start_epoch>=int(stage_epochs[stages[start_stage]]):start_stage+=1;start_epoch=0
        torch.set_rng_state(saved['torch_rng'].cpu());np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
        if saved.get('cuda_rng') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    log_path=run_dir/'train_metrics.jsonl'
    if bool(tc.get('cache_gpu_memory',False)):
        print('FARE6_MEMORY_CACHE '+json.dumps(memory.enable_gpu_cache(device,
                float(tc.get('gpu_cache_budget_gb',4.0)))),flush=True)
    loader=make_loader(train_ds,int(tc['batch_size']),shuffle=True,workers=int(tc.get('workers',0)))
    # Train-only marginal observation-sensitivity labels; never use Val/Test labels.
    support_file=run_dir/'support_teacher.pt'
    if int(cfg['teacher'].get('support_queries',0))>0:
        if support_file.exists():
            support_teacher=torch.load(support_file,map_location='cpu',weights_only=False)
        else:
            support_teacher=build_support_teacher(model,train_ds,device,
                   cfg['teacher']['support_queries'],cfg['experiment']['seed'],
                   cfg['teacher'].get('support_batch_size',8))
            torch.save(support_teacher,support_file)
    else:support_teacher={}
    teacher={}
    for stage_idx in range(start_stage,len(stages)):
        stage=stages[stage_idx];_set_stage(model,stage)
        for epoch in range(start_epoch if stage_idx==start_stage else 0,int(stage_epochs[stage])):
            epoch_started=time.perf_counter()
            if stage in ('retrieval','utility','risk','joint','calibration'):
                refresh=epoch==0 or epoch%max(1,int(cfg['teacher'].get('refresh_every_epochs',3)))==0
                file=run_dir/f'teacher_{stage}.pt'
                if refresh or not file.exists():
                    model.eval()
                    teacher=generate_utility_teacher(model,memory,train_ds,device,
                             cfg['teacher']['max_queries'],cfg['teacher']['batch_size'],
                             cfg['memory']['purge_steps'],cfg['experiment']['seed']+steps,
                             cfg['teacher'].get('nodes_per_query',16),
                             cfg['teacher'].get('candidates_per_query',6))
                    torch.save({'records':teacher,'metadata':{'stage':stage,'seed':cfg['experiment']['seed'],
                                                'rows':len(teacher),'step':steps}},file)
                    print('FARE5_TEACHER_REFRESH '+json.dumps({'stage':stage,'rows':len(teacher)}),flush=True)
                elif not teacher:
                    teacher=torch.load(file,map_location='cpu',weights_only=False)['records']
            model.train();total=torch.zeros((),device=device);count=0
            teacher_stats={'teacher_examples':0};epoch_replays=0;epoch_utility_labels=0
            epoch_utility_max=0.0;training_start=time.perf_counter()
            for raw in loader:
                batch=device_batch(raw,device);opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp):
                    out=model(batch,memory,purge=cfg['memory']['purge_steps'],return_details=False,
                              intervention='support_only' if stage=='support' else 'full')
                    loss,diag=forecasting_loss(out,batch,cfg)
                    if support_teacher:
                        sup_loss,sup_diag=support_auxiliary_loss(out,batch,support_teacher,device)
                        loss=loss+float(tc.get('support_teacher_weight',.05))*sup_loss
                        diag.update(sup_diag)
                    if teacher and stage in ('retrieval','utility','risk','joint','calibration') and steps%int(tc.get('replay_interval',8))==0:
                        # Rotate replay labels so supervision eventually covers the Teacher cache.
                        keys=list(teacher)
                        idx=(steps//max(1,int(tc.get('replay_interval',8))))*int(tc.get('replay_batch',2))
                        selected={keys[(idx+i)%len(keys)]:teacher[keys[(idx+i)%len(keys)]]
                                  for i in range(min(len(keys),int(tc.get('replay_batch',2))))} if keys else {}
                        aux,teacher_stats=replay_utility_loss(model,selected,device,len(selected),memory)
                        loss=loss+float(tc.get('utility_weight',.25))*aux
                        epoch_replays+=int(teacher_stats.get('utility_records',0))
                        epoch_utility_labels+=int(teacher_stats.get('utility_labels',0))
                        epoch_utility_max=max(epoch_utility_max,abs(float(aux.detach())))
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite FARE training loss')
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],float(tc['grad_clip']))
                scaler.step(opt);scaler.update()
                steps+=1;total+=loss.detach().float();count+=1
                if steps%int(tc.get('log_every',50))==0:
                    print('FARE6_TRAIN '+json.dumps({'stage':stage,'epoch':epoch+1,'step':steps,
                           'loss':float(loss.detach()),'mae':float(diag['mae']),
                           'alpha':float(out['alpha'].mean().detach()),
                           'null':float(out['selection']['null'].mean().detach()),**teacher_stats}),flush=True)
            train_sec=time.perf_counter()-training_start
            val=validate_loss(model,memory,val_ds,device,cfg,mode='support_only' if stage=='support' else 'full')
            elapsed_sec=time.perf_counter()-epoch_started
            state={'stage_idx':stage_idx,'stage':stage,'stage_epoch':epoch,'best_val':best,'steps':steps}
            metrics={'time':time.strftime('%Y-%m-%dT%H:%M:%S'),'stage':stage,'epoch':epoch+1,
                     'train_loss':float(total/max(1,count)),'val_mae_norm':val,'steps':steps,
                     'train_seconds':round(train_sec,2),'epoch_seconds':round(elapsed_sec,2),
                     'teacher_replays':epoch_replays,'teacher_labels':epoch_utility_labels,
                     'max_utility_loss':epoch_utility_max,**teacher_stats}
            if val<best and stage not in ('support','utility','retrieval'):
                best=val;state['best_val']=best
                checkpoint(run_dir/'best.pt',model,opt,cfg,state,metadata,scaler)
            checkpoint(run_dir/'last.pt',model,opt,cfg,state,metadata,scaler)
            stage_best_path=run_dir/('stage_'+stage+'_best.pt')
            stage_record=run_dir/('stage_'+stage+'_best.json')
            previous=json.loads(stage_record.read_text())['val_mae_norm'] if stage_record.exists() else float('inf')
            if val<previous:
                checkpoint(stage_best_path,model,opt,cfg,state,metadata,scaler)
                dump_json(stage_record,{'val_mae_norm':val,'epoch':epoch+1,'stage':stage})
            with log_path.open('a') as f:f.write(json.dumps(metrics)+'\n')
            print('FARE6_EPOCH '+json.dumps(metrics),flush=True)
        if cfg.get('innovation_audit',{}).get('stage_enabled',False):
            from .innovation_audit import audit
            audit(model,memory,val_ds,device,cfg,run_dir/'innovation_audit'/('stage_'+stage),stage=stage)
    if not (run_dir/'best.pt').exists():
        raise RuntimeError('No final-stage checkpoint. Include at least one Joint/Risk/Calibration stage')
    dump_json(run_dir/'training_complete.json',{'best_val_normalized':best,'steps':steps})
    print('FARE6_TRAIN_COMPLETE',flush=True)
    return run_dir/'best.pt'

@torch.no_grad()
def validate_loss(model,memory,ds,device,cfg,mode='full'):
    model.eval();num=0;den=0
    for raw in make_loader(ds,cfg['training']['batch_size'],workers=cfg['training']['workers']):
        batch=device_batch(raw,device)
        out=model(batch,memory,purge=cfg['memory']['purge_steps'],return_details=False,intervention=mode)
        v=batch['future_valid'].bool()
        num+=float(((out['prediction']-batch['target']).abs()*v).sum())
        den+=int(v.sum())
    return num/max(den,1)
