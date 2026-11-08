"""Full formal TCRR v4 training; S0 adapter -> caches -> S2/S3/S4 stages.

Uses existing Base_SupportBench split and retrieval/forecast cache. Test labels
are never loaded during model selection. All training labels are train-only.
"""
from __future__ import annotations
import copy,json,random,time,os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from .model import TCRRSTLAV4
from .cache import FrozenBaseCache,digest_model
from .losses import compute_v4_losses
from ..config import file_sha256,cache_path
from ..stla import load_pretrained_stla,MaskAwareSTLAformer
from ..data import SupportSplit,RetrievalCache
from ..evidence import FutureMemory
from ..replay_cache import prepare_bank
from ..normal_teacher import prepare_normal_teacher_bank
from ..trainer import seed_all,_make_loader,_batch_to_device,evidence_for_batch,predict
from ..traffic_neighbors import train_only_neighbours
from ..evaluation import metrics,save_summary
from ..checkpoint import persist_epoch,restore_training,atomic_save

def adapter_data_identity(cfg):
    """S0 depends on fixed formal observed-input and targets, not ckpt alone."""
    root=cache_path(cfg,'formal')
    names=('train_x_obs_norm.npy','val_x_obs_norm.npy',
           'train_history_observed.npy','train_artificial_missing.npy',
           'train_y_norm.npy','val_history_observed.npy','val_artificial_missing.npy','val_y_norm.npy')
    return {name:file_sha256(root/name) for name in names}

def initial_model(cfg,checkpoint,device,adapter_path=None):
    backbone,bc=load_pretrained_stla(Path(cfg['project']['root']),Path(checkpoint),device)
    teacher=MaskAwareSTLAformer(backbone).to(device)
    if adapter_path:
        payload=torch.load(adapter_path,map_location='cpu',weights_only=False)
        if payload.get('backbone_sha256')!=file_sha256(checkpoint):
            raise ValueError('Adapter trained on a different Backbone checkpoint')
        if payload.get('target')!=cfg['experiment']['target'] or payload.get('dataset')!=cfg['experiment']['dataset']:
            raise ValueError('Adapter target/dataset mismatch')
        if payload.get('data_identity')!=adapter_data_identity(cfg):
            raise ValueError('S0 adapter data/mask protocol fingerprint mismatch')
        teacher.mask_proj.load_state_dict(payload['mask_proj'],strict=True)
    return teacher,bc

def fit_shared_adapter(cfg,checkpoint,cache,output,device,smoke=False):
    """S0 once per dataset/target/backbone; returns fixed adapter checkpoint."""
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.is_file():
        initial_model(cfg,checkpoint,device,output)
        print(f'S0_REUSE adapter={output}',flush=True)
        return output
    teacher,bc=initial_model(cfg,checkpoint,device)
    for p in teacher.base.parameters():p.requires_grad_(False)
    for p in teacher.mask_proj.parameters():p.requires_grad_(True)
    train_ds=SupportSplit(cache,'train',24 if smoke else None)
    val_ds=SupportSplit(cache,'val',24 if smoke else None)
    bs=3 if smoke else int(cfg['training']['batch_size'])
    tl=_make_loader(train_ds,bs,True,cfg,smoke)
    vl=_make_loader(val_ds,bs,False,cfg,smoke)
    optimizer=torch.optim.AdamW(teacher.mask_proj.parameters(),lr=float(cfg.get('v4',{}).get('s0_lr',.0003)),weight_decay=.0001)
    scaler=torch.amp.GradScaler('cuda',enabled=(device.type=='cuda' and bool(cfg['runtime'].get('amp',True))))
    best_state=copy.deepcopy(teacher.mask_proj.state_dict())
    epochs=1 if smoke else int(cfg.get('v4',{}).get('s0_epochs',8))
    mean=float(bc['mean']);std=float(bc['std'])
    # Evaluate untouched, zero-initialized Mask Adapter before changing it.
    # If S0 does not improve validation, preserve the original STLA forecast.
    teacher.eval();initial_error=0.;initial_count=0.
    with torch.inference_mode():
        for vb in vl:
            vb=_batch_to_device(vb,device)
            pred,_=teacher(vb['x'],vb['mask'],vb['tod'],vb['dow'])
            v=vb['future_valid'].bool()
            initial_error+=float(((pred.float()*std+mean-vb['y_raw']).abs()*v).sum())
            initial_count+=int(v.sum())
    best=initial_error/max(initial_count,1)
    print(f'S0_BASELINE_VAL_MAE={best:.6f}',flush=True)
    for ep in range(1,epochs+1):
        for training,loader in ((True,tl),(False,vl)):
            teacher.train(training);teacher.base.eval()
            total=0.;count=0
            context=torch.enable_grad() if training else torch.no_grad()
            with context:
                for b in loader:
                    b=_batch_to_device(b,device)
                    if training:optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=scaler.is_enabled()):
                        pred,_=teacher(b['x'],b['mask'],b['tod'],b['dow'])
                        v=b['future_valid'].bool()
                        loss=((pred.float()*std+mean-b['y_raw'].float()).abs()*v).sum()/v.sum().clamp_min(1)
                    if training:
                        if scaler.is_enabled():
                            scaler.scale(loss).backward();scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(teacher.mask_proj.parameters(),5.)
                            scaler.step(optimizer);scaler.update()
                        else:loss.backward();nn.utils.clip_grad_norm_(teacher.mask_proj.parameters(),5.);optimizer.step()
                    total+=float(loss.detach())*int(v.sum());count+=int(v.sum())
            avg=total/max(count,1)
            if not training and avg<best:
                best=avg;best_state=copy.deepcopy(teacher.mask_proj.state_dict())
            print(f'S0_EPOCH ep={ep} phase={"train" if training else "val"} MAE={avg:.6f} best={best:.6f}',flush=True)
    payload=dict(schema=4,version=cfg['version'],target=cfg['experiment']['target'],dataset=cfg['experiment']['dataset'],
                 backbone_sha256=file_sha256(checkpoint),best_val_mae=best,mask_proj=best_state,
                 data_identity=adapter_data_identity(cfg))
    tmp=output.with_suffix('.partial');torch.save(payload,tmp);os.replace(tmp,output)
    print(f'S0_PASS adapter={output} best_val_MAE={best:.6f}',flush=True)
    return output

def build_v4_model(cfg,checkpoint,device,adapter_path):
    stla,bc=initial_model(cfg,checkpoint,device,adapter_path)
    neighbors=None
    train_cache=cache_path(cfg,'formal')
    if (train_cache/'train_x_full_norm.npy').is_file():
        neighbors=train_only_neighbours(train_cache,int(cfg.get('traffic',{}).get('graph_neighbors',8)))
    model=TCRRSTLAV4(stla,cfg,neighbors).to(device)
    model.freeze_base()
    return model,bc

def _run_epoch(model,loader,retrieval_cache,memory,device,mean,std,cfg,mode,stage,
               shared,bank,normal_bank,teacher,opt=None,scaler=None,epoch=0):
    training=opt is not None
    model.train(training)
    total={};n=0;valid_error=0.;valid_count=0.
    with (torch.enable_grad() if training else torch.no_grad()):
        for step,raw in enumerate(loader,1):
            b=_batch_to_device(raw,device)
            if training:opt.zero_grad(set_to_none=True)
            ev=evidence_for_batch(model,b,retrieval_cache,memory,device,cfg,mode,
                                  bank=bank,teacher=teacher,normal_bank=normal_bank)
            base,context=shared.fetch(b['index'],device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,
                                enabled=(device.type=='cuda' and bool(cfg['runtime'].get('amp',True)))):
                out=model(b['x'],b['mask'],b['tod'],b['dow'],mode=mode,
                          base_prediction=base,frozen_context=context,**ev)
                loss,stats=compute_v4_losses(out,b,mean,std,cfg,stage)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward();scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],float(cfg['training']['grad_clip']))
                    scaler.step(opt);scaler.update()
                else:
                    loss.backward();nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],float(cfg['training']['grad_clip']));opt.step()
            s=len(b['x']);n+=s
            for k,v in stats.items():total[k]=total.get(k,0.)+float(v)*s
            v=b['future_valid'].bool()
            valid_error+=float(((((out['prediction'].float()*std+mean)-b['y_raw']).abs()*v).sum()).detach())
            valid_count+=float(v.sum())
            if training and step%int(cfg['runtime'].get('print_every',100))==0:
                print(f'V4_TRAIN epoch={epoch} step={step}/{len(loader)} phase={stage} MAE={valid_error/max(valid_count,1):.6f}',flush=True)
    ans={k:x/max(1,n) for k,x in total.items()}
    ans['mae']=valid_error/max(valid_count,1)
    return ans

def stage_plan(cfg,smoke):
    s=cfg['v4']['stages']
    seq=[('transfer',int(s['transfer'])),('controller',int(s['controller'])),('joint',int(s['joint']))]
    if any(n<=0 for _,n in seq):raise ValueError('All three stages must have positive epochs')
    if sum(n for _,n in seq)!=int(cfg['training']['max_epochs']):
        raise ValueError('v4 stage sum must equal max_epochs')
    if smoke: return [('transfer',1),('controller',1),('joint',1)]
    return seq

def train_v4(cfg,cache,checkpoint,retrieval_train,retrieval_val,run_dir,adapter_path,
             smoke=False,resume=None):
    seed_all(int(cfg['experiment']['seed']))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,bc=build_v4_model(cfg,checkpoint,device,adapter_path)
    mean=float(bc['mean']);std=float(bc['std'])
    ds_train=SupportSplit(cache,'train',24 if smoke else None)
    ds_val=SupportSplit(cache,'val',24 if smoke else None)
    memory=FutureMemory(cache,256 if smoke else None)
    if memory._source_masks is None:raise RuntimeError('v4 requires train original validity masks')
    rt=RetrievalCache(retrieval_train);rv=RetrievalCache(retrieval_val)
    bs=3 if smoke else int(cfg['training']['batch_size'])
    loader_train=_make_loader(ds_train,bs,True,cfg,smoke)
    loader_val=_make_loader(ds_val,bs,False,cfg,smoke)
    teacher=copy.deepcopy(model.stla).to(device).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    cfg['runtime']['teacher_fingerprint']=digest_model(teacher)
    normal=prepare_normal_teacher_bank(cfg,memory,teacher,checkpoint,device,smoke=smoke)
    bank_train=prepare_bank(cfg,'train',ds_train,rt,memory,teacher,checkpoint,retrieval_train,device,smoke=smoke)
    bank_val=prepare_bank(cfg,'val',ds_val,rv,memory,teacher,checkpoint,retrieval_val,device,smoke=smoke)
    folder=Path(cfg['project']['retrieval_root'])/cfg['experiment']['dataset']/'v4'/('smoke' if smoke else 'formal')/'shared_base'
    shared_train=FrozenBaseCache(folder,model,ds_train,'train',checkpoint,adapter_path,device,
                                 build_batch_size=int(cfg['runtime'].get('base_cache_batch_size',32)))
    shared_val=FrozenBaseCache(folder,model,ds_val,'val',checkpoint,adapter_path,device,
                               build_batch_size=int(cfg['runtime'].get('base_cache_batch_size',32)))
    opt=torch.optim.AdamW([p for name,p in model.named_parameters() if not name.startswith('stla.')],
                          lr=float(cfg['training']['lr_new']),weight_decay=float(cfg['training']['weight_decay']))
    scaler=torch.amp.GradScaler('cuda',enabled=(device.type=='cuda' and bool(cfg['runtime'].get('amp',True))))
    run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=True)
    best=float('inf');best_epoch=0;history=[];stale=0;previous=None;start_epoch=1
    if resume:
        state=restore_training(resume,model,opt,scaler,cfg,checkpoint)
        best=float(state['best_val_mae']);best_epoch=int(state['best_epoch'])
        history=list(state['history']);stale=int(state['stale']);previous=state['previous_phase'];start_epoch=state['epoch']+1
    plan=stage_plan(cfg,smoke)
    stages=[label for label,amount in plan for _ in range(amount)]
    print(f'V4_TRAIN_PLAN epochs={len(stages)} stages={plan} Q={len(ds_train)} K={rt.candidate_count} mode={cfg["training"]["ablation"]}',flush=True)
    for epoch in range(start_epoch,len(stages)+1):
        phase=stages[epoch-1]
        model.set_stage(phase)
        if previous!=phase:
            if phase=='joint':
                # Production best must be judged with FULL C1/C2/C3 forward.
                best=float('inf');best_epoch=0
            stale=0;previous=phase
            print(f'V4_STAGE_START epoch={epoch} phase={phase} trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)}',flush=True)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        tick=time.time()
        train=_run_epoch(model,loader_train,rt,memory,device,mean,std,cfg,cfg['training']['ablation'],phase,
                         shared_train,bank_train,normal,teacher,opt,scaler,epoch)
        mid=time.time()
        val=_run_epoch(model,loader_val,rv,memory,device,mean,std,cfg,cfg['training']['ablation'],phase,
                       shared_val,bank_val,normal,teacher,epoch=epoch)
        rec=dict(epoch=epoch,phase=phase,
                 **{'train_'+k:v for k,v in train.items()},**{'val_'+k:v for k,v in val.items()},
                 seconds=time.time()-tick,train_seconds=mid-tick,
                 train_samples_per_second=len(ds_train)/max(mid-tick,1e-6),
                 peak_gpu_gib=(torch.cuda.max_memory_reserved()/1024**3 if device.type=='cuda' else 0))
        history.append(rec);print('V4_EPOCH '+json.dumps(rec),flush=True)
        improved=(val['mae']<best-float(cfg['training'].get('min_delta',1e-8)))
        if improved:best=float(val['mae']);best_epoch=epoch;stale=0
        else:stale+=1
        pd.DataFrame(history).to_csv(run_dir/'history.csv',index=False)
        persist_epoch(run_dir,model,opt,scaler,cfg,checkpoint,bc,'TCRR-STLA-v4',epoch,best,best_epoch,
                      history,stale,phase,improved)
        # Patience never skips a later stage; early stop applies only to joint.
        if phase=='joint' and stale>=int(cfg['training'].get('patience',20)):
            print(f'V4_EARLY_STOP epoch={epoch} best={best_epoch}',flush=True);break
    saved=torch.load(run_dir/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(saved['state_dict'],strict=True)
    model.set_stage('joint')
    from ..trainer import predict
    p=predict(model,loader_val,rv,memory,device,mean,std,cfg,cfg['training']['ablation'],bank_val,teacher,normal)
    # Calibrate on a held-out tail of the chronological VALIDATION split,
    # never on the test set. 1.0 is the unmodified model (no hard rejection).
    threshold,calibration=calibrate_rejection(p)
    model.v4.reject_threshold=threshold
    ck=saved
    ck['calibration']={'reject_threshold':threshold,**calibration}
    ck['adapter_sha256']=file_sha256(adapter_path)
    atomic_save(ck,run_dir/'best.pt')
    if threshold<1.0:
        p=predict(model,loader_val,rv,memory,device,mean,std,cfg,cfg['training']['ablation'],bank_val,teacher,normal)
    vm=metrics(p['prediction'],p['true'],p['valid'],p['direct'])
    bm=metrics(p['base'],p['true'],p['valid'],p['direct'])
    np.savez_compressed(run_dir/'val_predictions.npz',prediction=p['prediction'].astype('float32'),base=p['base'].astype('float32'))
    save_summary(run_dir/'summary.json',dict(model='TCRR-STLA-v4',best_epoch=best_epoch,
                 validation=vm,base_validation=bm,gate_mean=p['gate_mean'],backbone_checkpoint=str(checkpoint),
                 adapter_checkpoint=str(adapter_path),calibration=calibration,test_evaluated=False,
                 note='No official test metrics until evaluation completes'))
    print(f'V4_TRAIN_PASS best_epoch={best_epoch} val_MAE={vm["MAE"]:.6f}',flush=True)
    return run_dir/'best.pt'


def calibrate_rejection(predictions):
    """Small fixed grid, last 20% of validation queries; no test labels."""
    if 'harm_probability' not in predictions:
        return 1.0,{'status':'no_harm_head','source':'validation_tail'}
    q=len(predictions['prediction'])
    i=max(0,int(q*0.8))
    y=predictions['true'][i:]
    v=predictions['valid'][i:].astype(bool)
    base=predictions['base'][i:]
    pred=predictions['prediction'][i:]
    harm=predictions['harm_probability'][i:]
    if not v.any():return 1.0,{'status':'empty_valid_tail'}
    trials=[]
    # A threshold=1.0 leaves all predictions unchanged.
    for tau in (1.0,0.9,0.8,0.7,0.6,0.5,0.4):
        adjusted=np.where(harm>tau,base,pred)
        mae=float(np.abs(adjusted-y)[v].mean())
        coverage=float((harm<=tau)[v].mean())
        trials.append((mae,tau,coverage))
    # Select min MAE; ties prefer less aggressive rejection.
    best=min(trials,key=lambda row:(round(row[0],7),-row[1]))
    print(f'V4_RISK_CALIBRATION tau={best[1]} val_tail_mae={best[0]:.6f} coverage={best[2]:.4f}',flush=True)
    return float(best[1]),{'status':'calibrated','source':'chronological_validation_last_20pct',
                          'tail_mae':float(best[0]),'tail_coverage':float(best[2]),
                          'grid':[{'threshold':tau,'mae':mae,'coverage':cov} for mae,tau,cov in trials]}
