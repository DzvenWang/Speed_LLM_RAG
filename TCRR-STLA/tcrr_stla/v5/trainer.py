"""From-scratch v5 evidence pipeline and switch-aware training.

All model-generated artifacts live under /v5/. Shared Backbone and original
Base_SupportBench data are allowed external dependencies; legacy TCRR outputs
are not read. Epoch/checkpoints use data/source fingerprint verification.
"""
from __future__ import annotations
import json,time,os,hashlib,random,resource
from pathlib import Path
import torch,numpy as np,pandas as pd
from torch import nn
from torch.utils.data import Subset
from .model import TCRRSTLAV5,MODES,THREE_SWITCH_MODES
from .losses import compute_losses
from ..config import file_sha256,cache_path
from ..stla import load_pretrained_stla,MaskAwareSTLAformer
from ..data import SupportSplit,RetrievalCache
from ..evidence import FutureMemory
from ..replay_cache import prepare_bank
from ..normal_teacher import prepare_normal_teacher_bank
from ..trainer import seed_all,_make_loader,_batch_to_device,evidence_for_batch,predict
from ..traffic_neighbors import train_only_neighbours
from ..checkpoint import atomic_save,checkpoint_core
from ..evaluation import metrics,save_summary
from .shared_adapter import fit_shared_adapter, initial_model, effective_training_batch_size, smoke_training_dataset
from .calibration import choose_rejection_thresholds,chronological_partitions
from .base_cache import FrozenBaseCache,digest_model



@torch.no_grad()
def calibrate_controller(model,vl_cal,rv,memory,device,mean,std,cfg,shared,bank,normal,teacher):
    model.eval(); model.c3.set_calibration(enabled=False)
    gathered={k:[] for k in ('base','proposal','gate','net','risk','y','valid')}
    for raw in vl_cal:
        b=_batch_to_device(raw,device)
        ev=evidence_for_batch(model,b,rv,memory,device,cfg,'M11',bank=bank,teacher=teacher,normal_bank=normal)
        base,context=shared.fetch(b['index'],device)
        o=model(b['x'],b['mask'],b['tod'],b['dow'],mode='M11',controller_mode='learned',
                base_prediction=base,frozen_context=context,**ev)
        # Dual component gates cannot be reconstructed from scalar
        # gate * original proposal. Calibrate the ACTUAL un-rejected output.
        tensors=dict(base=o['base_prediction'].float()*std+mean,
                     proposal=o['correction'].float()*std,gate=torch.ones_like(o['gate']),
                     net=o['estimated_net_utility'],risk=o['harm_probability'],
                     y=b['y_raw'],valid=b['future_valid'])
        # Systematic subsampling keeps calibration memory bounded on Net-06;
        # same selection for base/proposal/labels/risks and independent of labels.
        stride=max(1,int(cfg['v5'].get('calibration_stride',16)))
        for key,value in tensors.items():
            gathered[key].append(value.detach().float().reshape(-1)[::stride].cpu())
    fields={key:torch.cat(value,dim=0) for key,value in gathered.items()}
    fitted=choose_rejection_thresholds(**{
        'base':fields['base'],'proposal':fields['proposal'],'gate':fields['gate'],
        'predicted_net':fields['net'],'harm_probability':fields['risk'],
        'y':fields['y'],'valid':fields['valid']},
        min_coverage=float(cfg['v5'].get('min_calibration_coverage',.55)))
    model.c3.set_calibration(fitted['min_utility'],fitted['max_harm_probability'],fitted['enabled'])
    print('V5_CALIBRATION '+json.dumps(fitted),flush=True)
    return fitted

def build_model(cfg,checkpoint,device,adapter_path):
    stla,bc=initial_model(cfg,checkpoint,device,adapter_path)
    neighbours=None
    root=cache_path(cfg,'formal')
    if (root/'train_x_full_norm.npy').is_file():
        neighbours=train_only_neighbours(root,int(cfg.get('traffic',{}).get('graph_neighbors',8)))
    model=TCRRSTLAV5(stla,cfg,neighbours).to(device)
    return model,bc


def stage_plan(cfg,smoke):
    v=cfg['v5']
    phases=[('c12',int(v.get('c12_epochs',50))),
            ('controller',int(v.get('controller_epochs',10))),
            ('joint',int(v.get('joint_epochs',0)))]
    phases=[(s,n) for s,n in phases if n>0]
    if smoke:return [('c12',1),('controller',1)]
    return phases


def run_epoch(model,loader,rc,memory,device,mean,std,cfg,phase,shared,bank,normal,teacher,
              optimizer=None,scaler=None,epoch=0):
    training=optimizer is not None
    model.train(training)
    totals={};samples=0;abs_sum=0.;sq_sum=0.;count=0
    start=time.perf_counter()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    active_validation_modes=(tuple(MODES) if phase=='c12' else tuple(THREE_SWITCH_MODES))
    switch_stats={mode:[0.,0] for mode in active_validation_modes}
    aux_weight=float(cfg['v5'].get('auxiliary_weight',.15))
    with (torch.enable_grad() if training else torch.no_grad()):
        for step,raw in enumerate(loader,1):
            b=_batch_to_device(raw,device)
            if training:optimizer.zero_grad(set_to_none=True)
            ev=evidence_for_batch(model,b,rc,memory,device,cfg,'M11',bank=bank,teacher=teacher,normal_bank=normal)
            base,context=shared.fetch(b['index'],device)
            amp=(device.type=='cuda' and bool(cfg['runtime'].get('amp',True)))
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                full_mode='M111' if phase in ('controller','joint') else 'M11'
                controller='auto' if phase in ('controller','joint') else 'fixed'
                if training:
                    o=model(b['x'],b['mask'],b['tod'],b['dow'],mode=full_mode,
                        controller_mode=controller,base_prediction=base,frozen_context=context,
                        shuffled_negatives=(phase in ('c12','joint') and
                            step%int(cfg['v5'].get('pair_interval',4))==0),**ev)
                    mode_outputs=None
                else:
                    mode_outputs=model.forward_modes(b['x'],b['mask'],b['tod'],b['dow'],
                        modes=active_validation_modes,controller_mode=controller,
                        base_prediction=base,frozen_context=context,**ev)
                    o=mode_outputs[full_mode]
                loss,stats=compute_losses(o,b,mean,std,cfg,phase)
                if training and aux_weight>0:
                    if phase=='c12':
                        m=('M00','M10','M01')[(step-1+epoch-1)%3]
                        aux_stage='c12'
                        aux_controller='fixed'
                    elif phase=='controller':
                        # Expose C3 to CF/Support off combinations too, so the
                        # eight-switch evaluation is not solely out-of-distribution.
                        m=('M001','M011','M101')[(step-1+epoch-1)%3]
                        aux_stage='controller'
                        aux_controller='auto'
                    else:
                        m=tuple(x for x in THREE_SWITCH_MODES if x!='M111')[(step-1+epoch-1)%7]
                        aux_stage='joint'
                        aux_controller='auto'
                    aux=model(b['x'],b['mask'],b['tod'],b['dow'],mode=m,
                              controller_mode=aux_controller,
                              base_prediction=base,frozen_context=context,**ev)
                    # C3 ON gets utility/risk supervision; OFF is trained on
                    # forecast loss only. Full mode is always trained above.
                    aux_is_c3_on=m in THREE_SWITCH_MODES and THREE_SWITCH_MODES[m][2]
                    aux_loss,_=compute_losses(aux,b,mean,std,cfg,aux_stage,
                                              auxiliary=not aux_is_c3_on)
                    loss=loss+aux_weight*aux_loss
                if training and phase in ('controller','joint') and float(cfg['v5'].get('c3_variant_aux_weight',.10))>0:
                    # The reference and restricted CRSI variants receive an
                    # actual forecast gradient on the same batch, not merely
                    # post-hoc OFF toggles on a Full-only trained controller.
                    variants=('scale','component','agreement','localized')
                    variant=variants[(step-1+epoch-1)%len(variants)]
                    alt=model.c3(o['base_prediction'],o['query_context'],o,
                        mode='learned',neighbours=model.neighbours,variant=variant)
                    alt_loss=((alt['prediction'].float()*std+mean-b['y_raw']).abs()
                              *b['future_valid']).sum()/b['future_valid'].sum().clamp_min(1)
                    loss=loss+float(cfg['v5'].get('c3_variant_aux_weight',.10))*alt_loss
            if training:
                params=[p for p in model.parameters() if p.requires_grad]
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward();scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(params,float(cfg['training'].get('grad_clip',5.)))
                    scaler.step(optimizer);scaler.update()
                else:
                    loss.backward();nn.utils.clip_grad_norm_(params,float(cfg['training'].get('grad_clip',5.)));optimizer.step()
            valid=b['future_valid'].bool()
            pred=(o['prediction'].detach().float()*std+mean)
            err=(pred-b['y_raw'].float())[valid]
            abs_sum+=float(err.abs().sum());sq_sum+=float(err.square().sum());count+=int(valid.sum())
            sz=len(b['x']);samples+=sz
            for k,v in stats.items():totals[k]=totals.get(k,0.)+float(v)*sz
            if not training:
                # All modes share identical query/retrieval/teacher evidence.
                # Before C3 training: four C1/C2 modes with fixed gate.
                # After C3 training: all eight C1/C2/C3 modes, C3 OFF=gate1.
                for m in active_validation_modes:
                    out=mode_outputs[m]
                    pe=(out['prediction'].float()*std+mean-b['y_raw'].float()).abs()
                    switch_stats[m][0]+=float((pe*valid).sum());switch_stats[m][1]+=int(valid.sum())
            if training and step%int(cfg['runtime'].get('print_every',100))==0:
                print(f'V5_STEP epoch={epoch} phase={phase} batch={step}/{len(loader)} MAE={abs_sum/max(count,1):.6f}',flush=True)
    result={k:v/max(samples,1) for k,v in totals.items()}
    result.update(mae=abs_sum/max(count,1),rmse=(sq_sum/max(count,1))**0.5)
    if device.type=='cuda':torch.cuda.synchronize(device)
    seconds=max(time.perf_counter()-start,1e-9)
    result['epoch_compute_seconds']=round(seconds,3)
    result['queries_per_second']=round(samples/seconds,2)
    # ru_maxrss on Linux is KiB. Maxima can include earlier stages in this process.
    result['peak_cpu_rss_gib']=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,3)
    if device.type=='cuda':
        result['peak_gpu_allocated_gib']=round(torch.cuda.max_memory_allocated(device)/1024**3,3)
        result['peak_gpu_reserved_gib']=round(torch.cuda.max_memory_reserved(device)/1024**3,3)
    if not training:
        result.update({f'{m}_mae':v[0]/max(v[1],1) for m,v in switch_stats.items()})
        if phase=='c12':
            result['cf_gain_without_support']=result['M00_mae']-result['M10_mae']
            result['cf_gain_with_support']=result['M01_mae']-result['M11_mae']
            result['support_gain_without_cf']=result['M00_mae']-result['M01_mae']
            result['support_gain_with_cf']=result['M10_mae']-result['M11_mae']
            result['cf_support_interaction']=(result['M10_mae']+result['M01_mae']-
                                               result['M00_mae']-result['M11_mae'])
        else:
            for cf in (0,1):
                for sp in (0,1):
                    result[f'c3_gain_cf{cf}_support{sp}']=(
                        result[f'M{cf}{sp}0_mae']-result[f'M{cf}{sp}1_mae'])
            for sp in (0,1):
                for c3 in (0,1):
                    result[f'cf_gain_support{sp}_c3{c3}']=(
                        result[f'M0{sp}{c3}_mae']-result[f'M1{sp}{c3}_mae'])
            for cf in (0,1):
                for c3 in (0,1):
                    result[f'support_gain_cf{cf}_c3{c3}']=(
                        result[f'M{cf}0{c3}_mae']-result[f'M{cf}1{c3}_mae'])
    return result


def train_v5(cfg,cache,checkpoint,retrieval_train,retrieval_val,run,adapter_path,smoke=False,resume=None):
    seed_all(int(cfg['experiment']['seed']))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,bc=build_model(cfg,checkpoint,device,adapter_path)
    mean=float(bc['mean']);std=float(bc['std'])
    train_ds=SupportSplit(cache,'train',24 if smoke else None)
    val_ds=SupportSplit(cache,'val',24 if smoke else None)
    partitions=chronological_partitions(val_ds.window_starts[:len(val_ds)],
        cfg['experiment']['history'],cfg['experiment']['horizon'],smoke=smoke)
    memory=FutureMemory(cache,256 if smoke else None)
    if memory._source_masks is None:raise RuntimeError('Requires original source validity masks')
    rt=RetrievalCache(retrieval_train);rv=RetrievalCache(retrieval_val)
    bs=effective_training_batch_size(cfg,smoke)
    tl=_make_loader(smoke_training_dataset(train_ds,cfg,smoke),bs,True,cfg,smoke)
    # Separate worker count: persistent validation loaders would otherwise
    # retain 3 more pools of workers and waste CPU/RAM during training.
    val_cfg={**cfg,'runtime':{**cfg['runtime'],
        'num_workers':int(cfg['runtime'].get('val_num_workers',0))}}
    vl=_make_loader(Subset(val_ds,partitions['selection']),bs,False,val_cfg,smoke)
    vl_cal=_make_loader(Subset(val_ds,partitions['calibration']),bs,False,val_cfg,smoke)
    vl_check=_make_loader(Subset(val_ds,partitions['check']),bs,False,val_cfg,smoke)
    # The STLA teacher is already frozen and kept in eval mode. Sharing the
    # SAME immutable module avoids one full duplicate model on V100 memory.
    teacher=model.stla.eval()
    for p in teacher.parameters():p.requires_grad_(False)
    cfg['runtime']['teacher_fingerprint']=digest_model(teacher)
    normal=prepare_normal_teacher_bank(cfg,memory,teacher,checkpoint,device,smoke=smoke)
    train_bank=prepare_bank(cfg,'train',train_ds,rt,memory,teacher,checkpoint,retrieval_train,device,smoke=smoke)
    val_bank=prepare_bank(cfg,'val',val_ds,rv,memory,teacher,checkpoint,retrieval_val,device,smoke=smoke)
    folder=(Path(cfg['project']['retrieval_root'])/cfg['experiment']['dataset']/'v5'/
          ('formal_smoke' if smoke else 'formal')/'shared_base')
    shared_train=FrozenBaseCache(folder,model,train_ds,'train',checkpoint,adapter_path,device,
                                 build_batch_size=int(cfg['runtime'].get('base_cache_batch_size',32)))
    shared_val=FrozenBaseCache(folder,model,val_ds,'val',checkpoint,adapter_path,device,
                               build_batch_size=int(cfg['runtime'].get('base_cache_batch_size',32)))
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    (run/'validation_partitions.json').write_text(json.dumps({
        **partitions,'selection_queries':len(partitions['selection']),
        'calibration_queries':len(partitions['calibration']),
        'check_queries':len(partitions['check'])},indent=2))
    print(f"V5_CHRONOLOGICAL_PARTITIONS selection={len(partitions['selection'])} "
          f"calibration={len(partitions['calibration'])} check={len(partitions['check'])}",flush=True)
    stages=[p for phase,n in stage_plan(cfg,smoke) for p in [phase]*n]
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
            lr=float(cfg['training']['lr_new']),weight_decay=float(cfg['training']['weight_decay']))
    scaler=torch.amp.GradScaler('cuda',enabled=(device.type=='cuda' and bool(cfg['runtime'].get('amp',True))))
    best={'c12':float('inf'),'controller':float('inf'),'joint':float('inf')}
    history=[];phase_last=None;start_epoch=1;resume_payload=None
    if resume:
        resume_payload=torch.load(resume,map_location='cpu',weights_only=False)
        if resume_payload.get('version')!=cfg['version'] or resume_payload.get('backbone_sha256')!=file_sha256(checkpoint):
            raise RuntimeError('Incompatible resume version or Backbone SHA')
        if resume_payload.get('adapter_sha256')!=file_sha256(adapter_path):
            raise RuntimeError('Incompatible resume Adapter SHA')
        prev=resume_payload['config']
        if prev['experiment']!=cfg['experiment'] or prev['retrieval']!=cfg['retrieval'] or prev['v5']!=cfg['v5']:
            raise RuntimeError('Incompatible resume experiment/retrieval/v5 configuration')
        model.load_state_dict(resume_payload['state_dict'],strict=True)
        start_epoch=int(resume_payload['epoch'])+1
        history=list(resume_payload['history'])
        for stage in best:
            candidate=[r['val_mae'] for r in history if r.get('stage')==stage]
            if candidate:best[stage]=min(candidate)
        random.setstate(resume_payload['rng_python'])
        np.random.set_state(resume_payload['rng_numpy'])
        torch.random.set_rng_state(resume_payload['rng_torch'])
        if torch.cuda.is_available() and resume_payload.get('rng_cuda') is not None:
            torch.cuda.set_rng_state_all(resume_payload['rng_cuda'])
        print(f'V5_RESUME_PASS epoch={start_epoch} previous_stage={resume_payload["stage"]}',flush=True)
    print(f'V5_PLAN plan={stage_plan(cfg,smoke)} K={rt.candidate_count} train={len(train_ds)} val={len(val_ds)} batch_size={bs} smoke_full_batch={bool(cfg["runtime"].get("smoke_full_batch", False))}',flush=True)
    for epoch in range(start_epoch,len(stages)+1):
        phase=stages[epoch-1]
        model.set_stage(phase)
        if phase!=phase_last:
            # A newly entered controller stage must start with best C12, not last C12.
            # Do not rewind when resuming an already-running controller stage.
            if phase=='controller' and (resume_payload is None or epoch!=start_epoch or
                                      resume_payload.get('stage')!='controller'):
                best_file=run/'best_c12.pt'
                if not best_file.is_file():raise RuntimeError('Missing best_c12.pt at controller transition')
                snapshot=torch.load(best_file,map_location='cpu',weights_only=False)
                model.load_state_dict(snapshot['state_dict'],strict=True)
                print(f'V5_RESTORE_BEST_C12 epoch={epoch} selected={snapshot["best_epoch"]}',flush=True)
            if phase=='joint' and (resume_payload is None or epoch!=start_epoch or
                                   resume_payload.get('stage')!='joint'):
                prior='best_controller.pt' if (run/'best_controller.pt').is_file() else 'best_c12.pt'
                snapshot=torch.load(run/prior,map_location='cpu',weights_only=False)
                model.load_state_dict(snapshot['state_dict'],strict=True)
                print(f'V5_RESTORE_BEFORE_JOINT source={prior}',flush=True)
            phase_last=phase
            # Reset optimizer per phase to avoid transferring stale momentum to newly unfrozen parameters.
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                    lr=float(cfg['training']['lr_new'])*(0.2 if phase=='joint' else 1.),
                    weight_decay=float(cfg['training']['weight_decay']))
            if resume_payload is not None and epoch==start_epoch and phase==resume_payload['stage']:
                optimizer.load_state_dict(resume_payload['optimizer_state_dict'])
                if resume_payload.get('scaler_state_dict') is not None:
                    scaler.load_state_dict(resume_payload['scaler_state_dict'])
            print(f'V5_STAGE_START phase={phase} epoch={epoch}',flush=True)
        t=time.time()
        train=run_epoch(model,tl,rt,memory,device,mean,std,cfg,phase,shared_train,train_bank,normal,teacher,
                        optimizer,scaler,epoch)
        val=run_epoch(model,vl,rv,memory,device,mean,std,cfg,phase,shared_val,val_bank,normal,teacher,epoch=epoch)
        rec={'epoch':epoch,'stage':phase,'seconds':time.time()-t,
             **{'train_'+k:v for k,v in train.items()},**{'val_'+k:v for k,v in val.items()}}
        history.append(rec)
        print('V5_EPOCH '+json.dumps(rec),flush=True)
        pd.DataFrame(history).to_csv(run/'history.csv',index=False)
        if val['mae']<best[phase]:
            best[phase]=val['mae']
            core=checkpoint_core(model,cfg,checkpoint,bc,'TCRR-STLA-v5',val['mae'],epoch)
            core.update(stage=phase,adapter_sha256=file_sha256(adapter_path))
            atomic_save(core,run/f'best_{phase}.pt')
        last=checkpoint_core(model,cfg,checkpoint,bc,'TCRR-STLA-v5',val['mae'],epoch)
        last.update(stage=phase,epoch=epoch,optimizer_state_dict=optimizer.state_dict(),
                    scaler_state_dict=scaler.state_dict(),history=history,adapter_sha256=file_sha256(adapter_path),
                    rng_python=random.getstate(),rng_numpy=np.random.get_state(),
                    rng_torch=torch.random.get_rng_state(),
                    rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        atomic_save(last,run/'last.pt')
    # Stage checkpoints have different forward-control policies. Evaluate each
    # consistently, then select the *lowest comparable validation MAE*.
    stage_candidates=[]
    for stage in ('c12','controller','joint'):
        path=run/f'best_{stage}.pt'
        if not path.is_file():continue
        payload=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(payload['state_dict'],strict=True)
        model.set_stage(stage)
        model.eval()
        cal=None
        if stage in ('controller','joint'):
            cal=calibrate_controller(model,vl_cal,rv,memory,device,mean,std,cfg,
                                     shared_val,val_bank,normal,teacher)
            # Update checkpoint with the calibrated C3 buffers; avoid overwriting
            # its original pre-calibration metric without retaining the original.
            payload['uncalibrated_best_val_mae']=payload['best_val_mae']
            payload['calibration']=cal
            payload['state_dict']={k:v.detach().cpu() for k,v in model.state_dict().items()}
            atomic_save(payload,path)
        value=run_epoch(model,vl,rv,memory,device,mean,std,cfg,stage,
                        shared_val,val_bank,normal,teacher,epoch=0)
        stage_candidates.append((float(value['mae']),stage,path,cal))
        print(f'V5_STAGE_VALIDATION stage={stage} MAE={value["mae"]:.6f}',flush=True)
    if not stage_candidates:raise RuntimeError('No stage checkpoint saved')
    best_mae,best_phase,best_path,cal=min(stage_candidates,key=lambda t:t[0])
    saved=torch.load(best_path,map_location='cpu',weights_only=False)
    model.load_state_dict(saved['state_dict'],strict=True)
    model.set_stage(best_phase)
    p=predict(model,vl,rv,memory,device,mean,std,cfg,'M11',bank=val_bank,
              teacher=teacher,normal_bank=normal)
    vm=metrics(p['prediction'],p['true'],p['valid'],p['direct'])
    bm=metrics(p['base'],p['true'],p['valid'],p['direct'])
    check_predictions=predict(model,vl_check,rv,memory,device,mean,std,cfg,'M11',
                              bank=val_bank,teacher=teacher,normal_bank=normal)
    heldout_check=metrics(check_predictions['prediction'],check_predictions['true'],
                          check_predictions['valid'],check_predictions['direct'])
    print('V5_LOCKED_CHECK '+json.dumps(heldout_check),flush=True)
    save_summary(run/'summary.json',dict(chronological_partition_file=str(run/'validation_partitions.json'),
                  locked_check=heldout_check,model='TCRR-STLA-v5',best_stage=best_phase,
                  selected_checkpoint=str(best_path),best_epoch=saved['best_epoch'],
                  validation=vm,base_validation=bm,
                  backbone_checkpoint=str(checkpoint),adapter_checkpoint=str(adapter_path),
                  calibration=cal,test_evaluated=False,
                  stage_selection=[dict(stage=stage,validation_mae=v,checkpoint=str(path))
                                   for v,stage,path,_ in stage_candidates],stage_best_mae=best))
    print(f'V5_TRAIN_PASS checkpoint={best_path} stage={best_phase} '
          f'val_MAE={vm["MAE"]:.6f}',flush=True)
    return best_path
