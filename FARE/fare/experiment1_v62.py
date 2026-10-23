"""FARE v6.2 Experiment 1: train support from scratch, then isolate Evidence Transfer.

No Test labels in any development stage. Historical OOF memory and the frozen
original STLAformer are the only reused artifacts from earlier experiments.
"""
from __future__ import annotations
import csv,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .config import dump_json,hash_file,cache_dir
from .protocol import SupportDataset,make_loader,device_batch
from .pipeline import prepare
from .candidate_bank_v62 import FrozenCandidateBank,build_candidate_bank
from .transfer_v62 import Experiment1Transfer,TransferEvidenceCache

VARIANTS=('direct','legacy','source_aligned','candidate_aligned')


def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def _base_and_support(model,batch):
    base=(model.backbone.encode_base(batch['x'],batch['mask'],batch['tod'],batch['dow'])
          if hasattr(model.backbone,'encode_base') else
          model.backbone(batch['x'],batch['mask'],batch['tod'],batch['dow']))
    hidden=model.base_proj(base['hidden'])
    support=model.support(batch['x'],batch['mask'],hidden,base['prediction'],model.graph,
                          topology=batch.get('topology'))
    pred=base['prediction']+model.support_delta(support['embedding']).squeeze(-1).permute(0,2,1)
    return base,support,pred


def _weighted_error(pred,batch):
    valid=batch['future_valid'].bool()
    err=(pred.float()-batch['target'].float())
    return (err.abs()*valid).sum()/valid.sum().clamp_min(1)


def _per_query_error(pred,batch):
    v=batch['future_valid'].float()
    e=pred.float()-batch['target'].float()
    den=v.sum((1,2)).clamp_min(1)
    return (e.abs()*v).sum((1,2))/den, (e.square()*v).sum((1,2))/den


def prepare_experiment(cfg,device,backbone_file,memory_file):
    model,memory,train,val,_,meta=prepare(cfg,device,backbone_file=backbone_file,
                                             memory_file=memory_file,build=False,cuda_safety=False)
    if memory.metadata.get('residual_provenance')!='external_oos_paired':
        raise RuntimeError('Experiment 1 requires verified paired OOF Memory')
    return model,memory,train,val,meta


def _save_checkpoint(path,model,extra,stage):
    path=Path(path)
    if path.exists():raise FileExistsError(f'Refusing to overwrite checkpoint: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'version':'6.2.0','stage':stage,'state_dict':model.state_dict(),
                'metadata':extra},path)


def _load_support(model,ckpt,meta):
    saved=torch.load(ckpt,map_location='cpu',weights_only=False)
    if saved.get('version')!='6.2.0' or saved.get('stage')!='support':
        raise RuntimeError('Not a same-run v6.2 Support-only checkpoint')
    for field in ('target','dataset','seed','backbone_sha256','memory_sha256'):
        if saved['metadata'].get(field)!=meta.get(field):
            raise RuntimeError(f'Support checkpoint provenance mismatch: {field}')
    model.load_state_dict(saved['state_dict'],strict=True)
    model.eval()
    for param in model.parameters():param.requires_grad_(False)
    return model


def _train_support(model,train,val,device,cfg,output,meta):
    out=Path(output); ck=out/'support_best.pt'
    if ck.exists():raise FileExistsError(f'Support already trained: {ck}')
    for param in model.parameters():param.requires_grad_(False)
    allowed=('base_proj.','support.','support_delta.')
    for name,param in model.named_parameters():
        if name.startswith(allowed):param.requires_grad_(True)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                lr=float(cfg['experiment1']['support_lr']),weight_decay=1e-4)
    epochs=int(cfg['experiment1']['support_epochs']);bs=int(cfg['experiment1']['support_batch_size'])
    workers=int(cfg['experiment1']['workers']);best=float('inf');best_state=None
    loader=make_loader(train,bs,shuffle=True,workers=workers)
    out.mkdir(parents=True,exist_ok=True)
    for epoch in range(epochs):
        started=time.perf_counter();model.train();loss_total=torch.zeros((),device=device);steps=0
        for raw in loader:
            batch=device_batch(raw,device)
            optimizer.zero_grad(set_to_none=True)
            _,_,pred=_base_and_support(model,batch)
            loss=_weighted_error(pred,batch)
            if not torch.isfinite(loss):raise FloatingPointError('Support loss nonfinite')
            loss.backward();nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),5.)
            optimizer.step();loss_total+=loss.detach();steps+=1
        model.eval();num=torch.zeros((),device=device);den=torch.zeros((),device=device)
        with torch.no_grad():
            for raw in make_loader(val,bs,workers=workers):
                batch=device_batch(raw,device)
                _,_,pred=_base_and_support(model,batch)
                valid=batch['future_valid'].bool()
                num+=(pred-batch['target']).abs().mul(valid).sum()
                den+=valid.sum()
        mae=float((num/den.clamp_min(1)).cpu())
        rec={'stage':'support','epoch':epoch+1,'val_mae_normalized':mae,
             'train_loss':float((loss_total/max(steps,1)).cpu()),
             'seconds':round(time.perf_counter()-started,2)}
        print('FARE62_SUPPORT_EPOCH '+json.dumps(rec),flush=True)
        with (out/'support_metrics.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
        if mae<best:
            best=mae
            # Snapshot CPU tensors; don't retain live mutable model state.
            best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    if best_state is None:raise RuntimeError('Support training produced no checkpoint')
    if ck.exists():raise FileExistsError(ck)
    torch.save({'version':'6.2.0','stage':'support','state_dict':best_state,'metadata':meta},ck)
    dump_json(out/'support_summary.json',{'best_val_mae_normalized':best,'checkpoint':str(ck),
               'checkpoint_sha256':hash_file(ck),'epochs':epochs,'fresh_fare_initialization':True,
               'backbone_checkpoint':meta.get('backbone_checkpoint'),
               'memory_path':meta.get('memory_path'),
               'backbone_sha256':meta.get('backbone_sha256'),
               'memory_sha256':meta.get('memory_sha256')})
    print('FARE62_SUPPORT_PASS '+str(ck),flush=True)
    return ck


def train_support(cfg,device,backbone,memory_path,experiment_dir):
    set_seed(cfg['experiment']['seed']);model,memory,train,val,meta=prepare_experiment(cfg,device,backbone,memory_path)
    return _train_support(model,train,val,device,cfg,Path(experiment_dir)/'support',meta)


def _pilot_datasets(cfg):
    c=cache_dir(cfg,'formal');e=cfg['experiment1']
    train=SupportDataset(c,'train',e.get('pilot_train_limit'),strategy='spread')
    val=SupportDataset(c,'val',e.get('pilot_val_limit'),strategy='spread')
    return train,val


def candidate_bank(cfg,device,backbone,memory_path,experiment_dir):
    set_seed(cfg['experiment']['seed']);model,memory,_,_,meta=prepare_experiment(cfg,device,backbone,memory_path)
    ck=Path(experiment_dir)/'support'/'support_best.pt'
    _load_support(model,ck,meta)
    train,val=_pilot_datasets(cfg)
    return build_candidate_bank(model,memory,model.graph,train,val,device,
             Path(experiment_dir)/'candidate_bank',cfg,hash_file(ck),meta,
             batch_size=int(cfg['experiment1']['candidate_batch_size']),
             workers=int(cfg['experiment1']['workers']))


def _run_transfer(model,head,bank,memory,batch,device):
    with torch.no_grad():base,support,support_pred=_base_and_support(model,batch)
    short=bank.fetch(batch['row_id'],device)
    return head(short,memory,model.graph,support,base['prediction'],
                reference_prediction=support_pred),support_pred


def _evaluate_transfer(model,head,memory,val,bank,device,cfg,trace_file=None):
    head.eval();model.eval();traces=[]
    stats=torch.zeros(10,device=device,dtype=torch.float64)
    bsz=int(cfg['experiment1']['transfer_batch_size'])
    for raw in make_loader(val,bsz,workers=int(cfg['experiment1']['workers'])):
        batch=device_batch(raw,device)
        with torch.no_grad():
            out,support_pred=_run_transfer(model,head,bank,memory,batch,device)
            pred=out['prediction']
            full_mae,full_mse=_per_query_error(pred,batch)
            sup_mae,sup_mse=_per_query_error(support_pred,batch)
            # Matched history-only identity shuffle within each query's legal pool.
            dv=out['candidate_deltas'];av=out['candidate_valid']
            rolled_dv=dv.roll(1,2);rolled_av=av.roll(1,2)
            w=out['candidate_weights']*rolled_av
            w=w/w.sum(2,keepdim=True).clamp_min(1e-8)
            shuffled=(w*rolled_dv).sum(2)
            alpha=out['learned_alpha'].permute(0,2,1)
            shuffled_pred=support_pred+alpha*shuffled.permute(0,2,1)
            shuffled_mae,_=_per_query_error(shuffled_pred,batch)
            # Diagnostic oracle uses true VAL labels but NEVER selects candidates
            # for reported deployed predictions or for any model updates.
            v=batch['future_valid'].permute(0,2,1).bool()
            tgt=batch['target'].permute(0,2,1)
            baseline=support_pred.permute(0,2,1)
            loss_base=(baseline-tgt).abs()
            oracle_err=loss_base.clone()
            for coeff in (0.1,0.25,0.5,1.):
                errors=(baseline[:,:,None,:]+coeff*dv-tgt[:,:,None,:]).abs()
                errors=errors.masked_fill(~av,1e9)
                oracle_err=torch.minimum(oracle_err,errors.min(2).values)
            oracle_mae=(oracle_err*v).sum((1,2))/v.sum((1,2)).clamp_min(1)
            for i in range(len(full_mae)):
                traces.append((int(batch['row_id'][i]),int(batch['origin'][i]),
                               float(full_mae[i]),float(full_mse[i]),float(sup_mae[i]),
                               float(sup_mse[i]),float(shuffled_mae[i]),float(oracle_mae[i])))
            directly_missing=batch['direct_mask'].bool()[:,None,:]
            masked_valid=batch['future_valid'].bool() & directly_missing
            masked_den=masked_valid.sum()
            masked_err=pred.float()-batch['target'].float()
            support_err=support_pred.float()-batch['target'].float()
            stats+=torch.stack([full_mae.sum(),full_mse.sum(),sup_mae.sum(),sup_mse.sum(),
                                shuffled_mae.sum(),oracle_mae.sum(),
                                (masked_err.abs()*masked_valid).sum(),
                                (masked_err.square()*masked_valid).sum(),
                                (support_err.abs()*masked_valid).sum(),masked_den]).double()
    if not traces:raise RuntimeError('Empty validation evaluation')
    # Original-unit errors: target = normalised using the frozen STLAformer
    # checkpoint's target mean/std, and affine inversion scales MAE/RMSE by std.
    target_std=float(model.backbone.std)
    if not np.isfinite(target_std) or target_std<=0:raise RuntimeError('Invalid backbone target std')
    a=np.array(traces,dtype=np.float64)
    if trace_file is not None:
        np.save(trace_file,a,allow_pickle=False)
    result={'samples':len(traces),
            'MAE':float(stats[0]/len(traces))*target_std,
            'RMSE':float((stats[1]/len(traces)).sqrt())*target_std,
            'support_MAE':float(stats[2]/len(traces))*target_std,
            'support_RMSE':float((stats[3]/len(traces)).sqrt())*target_std,
            'shuffled_MAE':float(stats[4]/len(traces))*target_std,
            'oracle_with_null_MAE':float(stats[5]/len(traces))*target_std,
            'masked_MAE':float(stats[6]/stats[9])*target_std if stats[9]>0 else None,
            'masked_RMSE':float((stats[7]/stats[9]).sqrt())*target_std if stats[9]>0 else None,
            'support_masked_MAE':float(stats[8]/stats[9])*target_std if stats[9]>0 else None,
            'mae_normalized':float(stats[0]/len(traces)),
            'rmse_normalized':float((stats[1]/len(traces)).sqrt()),
            'support_mae_normalized':float(stats[2]/len(traces)),
            'support_rmse_normalized':float((stats[3]/len(traces)).sqrt()),
            'shuffled_mae_normalized':float(stats[4]/len(traces)),
            'oracle_with_null_mae_normalized':float(stats[5]/len(traces)),
            'evidence_benefit':float((stats[2]-stats[0])/len(traces)),
            'identity_benefit':float((stats[4]-stats[0])/len(traces))}
    return result


def direct_feasibility(cfg,device,backbone,memory_path,experiment_dir):
    """No Transfer training: fixed-candidate direct-residual oracle + NULL.

    Candidate oracle consumes VAL future for research diagnosis ONLY, never for
    ranking, checkpoint selection, or deployed test prediction.
    """
    set_seed(cfg['experiment']['seed'])
    model,memory,_,_,meta=prepare_experiment(cfg,device,backbone,memory_path)
    root=Path(experiment_dir)
    support_ck=root/'support'/'support_best.pt'
    _load_support(model,support_ck,meta)
    _,val=_pilot_datasets(cfg)
    bank=FrozenCandidateBank(root/'candidate_bank','val',val,
           {'support_checkpoint_sha256':hash_file(support_ck),
            'backbone_sha256':meta['backbone_sha256'],
            'memory_sha256':meta['memory_sha256']})
    head=Experiment1Transfer('direct',d=int(cfg['model']['hidden_dim']),
            horizon=int(cfg['model']['horizon']),temperature=float(cfg['experiment1']['candidate_temperature'])).to(device)
    report=_evaluate_transfer(model,head,memory,val,bank,device,cfg)
    report.update({'trained_transfer':False,'transfer_variant':'direct_raw',
                   'uses_val_future_for_deployed_prediction':False,
                   'candidate_oracle_is_diagnostic_only':True})
    if (root/'direct_feasibility.json').exists():raise FileExistsError('Direct feasibility already exists')
    dump_json(root/'direct_feasibility.json',report)
    print('FARE62_DIRECT_FEASIBILITY_PASS '+json.dumps(report),flush=True)
    return report


def train_transfer(cfg,device,backbone,memory_path,experiment_dir,variant):
    if variant not in VARIANTS:raise ValueError(variant)
    set_seed(cfg['experiment']['seed']);model,memory,_,_,meta=prepare_experiment(cfg,device,backbone,memory_path)
    exdir=Path(experiment_dir);support_ck=exdir/'support'/'support_best.pt'
    _load_support(model,support_ck,meta)
    pilot_train,pilot_val=_pilot_datasets(cfg)
    bank_root=exdir/'candidate_bank'
    expected={'support_checkpoint_sha256':hash_file(support_ck),
              'backbone_sha256':meta['backbone_sha256'],'memory_sha256':meta['memory_sha256'],
              'target':meta['target'],'dataset':meta['dataset']}
    train_bank=FrozenCandidateBank(bank_root,'train',pilot_train,expected)
    val_bank=FrozenCandidateBank(bank_root,'val',pilot_val,expected)
    TransferEvidenceCache.attach(memory)
    set_seed(int(cfg['experiment']['seed'])+777)
    head=Experiment1Transfer(variant,d=int(cfg['model']['hidden_dim']),
                             horizon=int(cfg['model']['horizon']),
                             max_lag=int(cfg['traffic']['max_temporal_lag']),
                             temperature=float(cfg['experiment1']['candidate_temperature'])).to(device)
    optim=torch.optim.AdamW(head.parameters(),lr=float(cfg['experiment1']['transfer_lr']),weight_decay=1e-4)
    outdir=exdir/'transfer'/variant
    ck=outdir/'best.pt'
    if ck.exists():raise FileExistsError(f'Refusing to overwrite variant {ck}')
    outdir.mkdir(parents=True,exist_ok=True)
    best=float('inf');best_state=None
    if bool(cfg['training'].get('cache_gpu_memory',False)):
        names=['residual_complete','residual_valid','future','future_valid','failure_delta','failure_valid',
               'x','mask','_v62_source_evolution','_v62_source_evolution_valid']
        if variant=='legacy':names.remove('_v62_source_evolution');names.remove('_v62_source_evolution_valid')
        if variant=='direct':names=['residual_complete','residual_valid']
        print('FARE62_GPU_CACHE '+json.dumps(memory.enable_gpu_cache(device,
             float(cfg['experiment1'].get('gpu_cache_budget_gb',5.0)),sources=names)),flush=True)
    loader=make_loader(pilot_train,int(cfg['experiment1']['transfer_batch_size']),shuffle=True,
                       workers=int(cfg['experiment1']['workers']))
    for epoch in range(int(cfg['experiment1']['transfer_epochs'])):
        head.train();total=torch.zeros((),device=device);steps=0;skipped_batches=0;started=time.perf_counter()
        for raw in loader:
            batch=device_batch(raw,device)
            # Early chronological windows may have no completed OOF histories.
            # Such batches cannot supervise Transfer and must not consume GPU steps.
            if not train_bank.fetch(batch['row_id'],device)['valid'].any():
                skipped_batches+=1
                continue
            optim.zero_grad(set_to_none=True)
            output,ref=_run_transfer(model,head,train_bank,memory,batch,device)
            loss=_weighted_error(output['prediction'],batch)
            if not torch.isfinite(loss):raise FloatingPointError(f'{variant}: nonfinite transfer loss')
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(),float(cfg['training'].get('grad_clip',5.0)))
            optim.step();total+=loss.detach();steps+=1
        report=_evaluate_transfer(model,head,memory,pilot_val,val_bank,device,cfg)
        report.update({'variant':variant,'epoch':epoch+1,'train_loss':float((total/max(steps,1)).cpu()),
                       'steps':steps,'no_evidence_batches_skipped':skipped_batches,
                       'seconds':round(time.perf_counter()-started,2)})
        if steps==0:raise RuntimeError('No legal OOF candidate in Transfer training; inspect candidate bank coverage')
        print('FARE62_TRANSFER_EPOCH '+json.dumps(report),flush=True)
        with (outdir/'metrics.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
        if report['mae_normalized']<best:
            best=report['mae_normalized']
            best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    if best_state is None:raise RuntimeError('No transfer checkpoint')
    metadata={**meta,'variant':variant,'support_checkpoint_sha256':hash_file(support_ck),
              'candidate_bank_manifest_sha256':hash_file(bank_root/'manifest.json'),
              'train_count':len(pilot_train),'val_count':len(pilot_val)}
    torch.save({'version':'6.2.0','stage':'experiment1_transfer',
                'state_dict':best_state,'metadata':metadata},ck)
    head.load_state_dict(best_state,strict=True)
    metrics=_evaluate_transfer(model,head,memory,pilot_val,val_bank,device,cfg,
                               trace_file=outdir/'val_query_errors.npy')
    metrics.update({'variant':variant,'checkpoint_sha256':hash_file(ck),
                    'checkpoint':str(ck),'support_checkpoint_sha256':hash_file(support_ck)})
    dump_json(outdir/'summary.json',metrics)
    print('FARE62_TRANSFER_PASS '+json.dumps(metrics),flush=True)
    return metrics


def _bootstrap_diff(target,control,origin,replicas=1000,seed=10):
    """Paired 288-step time-block bootstrap; absent blocks fall back to queries."""
    rng=np.random.default_rng(seed)
    keys=(origin.astype(np.int64)//288)
    blocks=np.unique(keys)
    by_block=[np.flatnonzero(keys==b) for b in blocks]
    diffs=np.asarray(control)-np.asarray(target)
    draws=[]
    for _ in range(replicas):
        chosen=rng.integers(len(blocks),size=len(blocks))
        idx=np.concatenate([by_block[i] for i in chosen])
        draws.append(float(diffs[idx].mean()))
    return {'mean_gain':float(diffs.mean()),'ci95':list(map(float,np.percentile(draws,[2.5,97.5]))),
            'time_blocks':len(blocks)}


def summarize_experiment(cfg,experiment_dir):
    exdir=Path(experiment_dir);reports={};tracks={}
    for variant in VARIANTS:
        path=exdir/'transfer'/variant
        reports[variant]=json.loads((path/'summary.json').read_text())
        tracks[variant]=np.load(path/'val_query_errors.npy',allow_pickle=False)
    ref=tracks[VARIANTS[0]]
    for name,t in tracks.items():
        if not np.array_equal(ref[:,:2],t[:,:2]):raise RuntimeError(f'Pairing mismatch for {name}')
    output=[];comparisons={}
    for name,t in tracks.items():
        outcome=reports[name]
        outcome['paired_benefit_over_support']=_bootstrap_diff(t[:,2],t[:,4],t[:,1],
                  seed=cfg['experiment']['seed'])
        outcome['paired_identity_benefit']=_bootstrap_diff(t[:,2],t[:,6],t[:,1],
                  seed=cfg['experiment']['seed']+1)
        output.append(outcome)
    best_simple=min(VARIANTS[:3],key=lambda name:reports[name]['mae_normalized'])
    new=tracks['candidate_aligned'];base=tracks[best_simple]
    paired=_bootstrap_diff(new[:,2],base[:,2],new[:,1],seed=cfg['experiment']['seed']+2)
    good_oracle=(new[:,4]-new[:,7]).mean()
    improvement=paired['mean_gain']
    # A pilot is a GO only with a positive paired CI, evidence identity gain,
    # and true-VAL-label diagnostic headroom. Never label C1/C2/C3 supported.
    identity=output[-1]['paired_identity_benefit']
    # A three-query smoke or tiny pilot must NEVER pass a statistical GO gate.
    min_reliable=int(cfg.get('experiment1',{}).get('min_audit_queries',512))
    passed=bool(len(new)>=min_reliable and paired['time_blocks']>=10 and
                paired['ci95'][0]>0 and identity['ci95'][0]>0 and good_oracle>0 and
                paired['mean_gain']>=0.003*max(1e-8,float(base[:,2].mean())))
    verdict={'experiment':'FARE_v6.2_Experiment1','decision':'GO_TO_EXPERIMENT2' if passed else 'INVESTIGATE_TRANSFER',
             'best_simple_transfer':best_simple,'T3_vs_best_simple':paired,
             'T3_identity_gain':identity,'oracle_with_null_headroom':float(good_oracle),
             'innovation_status':{'C1':'INCONCLUSIVE','C2':'INCONCLUSIVE','C3':'INCONCLUSIVE'},
             'validation_only':True,'paired_block_bootstrap':True,
             'audited_queries':len(new),'minimum_queries_for_go':min_reliable,
             'notes':'Oracle uses validation future only for diagnosis, never deployed prediction'}
    dump_json(exdir/'experiment1_verdict.json',verdict)
    with (exdir/'experiment1_table.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['variant','MAE','RMSE','masked_MAE','masked_RMSE',
              'support_MAE','support_RMSE','support_masked_MAE','shuffled_MAE',
              'oracle_with_null_MAE','mae_normalized','rmse_normalized',
              'support_mae_normalized','oracle_with_null_mae_normalized','shuffled_mae_normalized',
              'evidence_benefit','identity_benefit']);w.writeheader()
        for r in output:w.writerow({k:r.get(k) for k in w.fieldnames})
    print('FARE62_EXPERIMENT1_VERDICT '+json.dumps(verdict),flush=True)
    return verdict
