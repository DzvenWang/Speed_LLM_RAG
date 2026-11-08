#!/usr/bin/env python3
"""One-shot four-mode evaluation using one shared teacher/retrieval per batch.

No large NPZ by default; results are small per-mode CSV and JSON summaries.
"""
from __future__ import annotations
import argparse,os,sys,json,re,hashlib,time,resource
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tcrr_stla.config import cache_path,relocate_saved_config,file_sha256
from tcrr_stla.data import SupportSplit,RetrievalCache
from tcrr_stla.evidence import FutureMemory
from tcrr_stla.normal_teacher import prepare_normal_teacher_bank
from tcrr_stla.replay_cache import prepare_bank
from tcrr_stla.evidence import counterfactual_residuals
from tcrr_stla.v5.trainer import build_model,_make_loader,_batch_to_device,evidence_for_batch
from tcrr_stla.v5.model import MODES,THREE_SWITCH_MODES,ALL_MODES
from tcrr_stla.v5.diagnostics import MechanismDiagnostics,select_matched_wrong_mask
from tcrr_stla.v5.p0_deep_diagnostics import P0DeepDiagnostics
from tcrr_stla.v5.calibration import chronological_partitions
from torch.utils.data import Subset
from tcrr_stla.v5.base_cache import digest_model
from tcrr_stla.evaluation import summarize_by_family,save_summary
from scripts.build_retrieval_cache import ensure_cache


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--backbone-checkpoint')
    p.add_argument('--adapter-checkpoint')
    p.add_argument('--split',choices=['validation','test'],default='test')
    p.add_argument('--validation-section',choices=['all','selection','calibration','check'],default='all')
    p.add_argument('--severities',nargs='+',type=int,default=[20,40,60,90])
    p.add_argument('--modes',nargs='+',choices=list(ALL_MODES),default=None)
    p.add_argument('--controller-mode',choices=['auto','fixed','learned','none'],default='auto')
    p.add_argument('--controller-variant',choices=['scale','component','agreement','localized','full'],default=None)
    p.add_argument('--disable-calibration',action='store_true',help='Compare C3 architecture without a Full-trained rejection threshold')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--save-predictions',action='store_true')
    p.add_argument('--cache-test-replay',action='store_true',
                   help='Opt in to on-disk CF Replay per test severity, useful for repeated test passes')
    p.add_argument('--disable-shared-modes',action='store_true',
                   help='Diagnostic fallback: evaluate each mechanism mode separately')
    p.add_argument('--max-batches',type=int,default=0,help='Diagnostic limit; 0 evaluates the full split')
    p.add_argument('--output-tag',default=None,help='Separate output directory under evaluation/<split>/automated/')
    p.add_argument('--diagnostics',action='store_true',help='Streaming C1/C2/C3 mechanism statistics')
    p.add_argument('--p0-deep',action='store_true',
                   help='Paired C1/C2/C3 candidate and position-level P0 diagnostics; requires M000 M100 M110 M111')
    p.add_argument('--cf-control',choices=['correct','shuffled','wrong_mask'],default='correct')
    p.add_argument('--support-control',choices=['correct','coverage','shuffled'],default='correct')
    a=p.parse_args()
    if a.max_batches < 0:
        p.error('--max-batches must be >= 0')
    if a.batch_size < 1:
        p.error('--batch-size must be >= 1')
    if a.output_tag is not None and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.output_tag):
        p.error('--output-tag must be a single safe directory component')
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if ck.get('version')!='5.3.0':raise RuntimeError('v5.3.0 checkpoint required')
    # A C12 checkpoint has no trained C3: keep legacy four-mode fixed-gate
    # diagnosis by default. Controller/joint defaults to all eight toggles.
    if a.modes is None:
        a.modes=list(MODES) if ck.get('stage','c12')=='c12' else list(THREE_SWITCH_MODES)
    if any(m in THREE_SWITCH_MODES for m in a.modes) and a.controller_mode!='auto':
        raise ValueError('Mxyz encodes C3; use --controller-mode auto. '
                         'For fixed/none override comparisons, use legacy Mxy.')
    if len(set(a.modes))!=len(a.modes):
        raise ValueError('Duplicate evaluation modes')
    if a.p0_deep:
        required={'M000','M100','M110','M111'}
        if not required.issubset(a.modes):
            raise ValueError('--p0-deep requires M000 M100 M110 M111 on one checkpoint')
        if ck.get('stage') not in ('controller','joint'):
            raise ValueError('--p0-deep requires a trained controller or joint checkpoint')
        if a.cf_control!='correct' or a.support_control!='correct':
            raise ValueError('--p0-deep requires correct CF/support; evaluate negative controls separately')
    cfg=relocate_saved_config(ck['config'])
    os.environ['TRAFFIDENT_TARGET']=cfg['experiment']['target']
    cfg['runtime'].update(replay_bank_on_gpu=False,normal_teacher_on_gpu=False)
    # Avoid worker process RAM duplication by default; power users can opt in
    # after measuring CPU RSS, without changing the cache or experiment data.
    cfg['runtime']['num_workers']=int(os.environ.get('TCRR_EVAL_WORKERS','0'))
    # IMPORTANT: this config is the formal *training* identity embedded in the
    # checkpoint. S0 checks the original training batch size (e.g. 64). Never
    # overwrite it with the evaluation DataLoader batch size (e.g. 8): doing
    # so makes a valid, SHA-matched S0 adapter fail its identity check.
    # The evaluation loader below already receives a.batch_size explicitly.
    print('V5_EVAL_BATCH_CONFIG '+json.dumps({
        'training_batch_size':int(cfg['training']['batch_size']),
        'evaluation_batch_size':int(a.batch_size),
        'evaluation_workers':int(cfg['runtime']['num_workers'])}), flush=True)
    backbone=Path(a.backbone_checkpoint or ck['backbone_checkpoint'])
    if file_sha256(backbone)!=ck['backbone_sha256']:raise RuntimeError('Backbone SHA mismatch')
    adapter=Path(a.adapter_checkpoint or Path(a.checkpoint).resolve().parent/'s0_shared_adapter.pt')
    if not adapter.is_file():raise FileNotFoundError(adapter)
    if file_sha256(adapter)!=ck['adapter_sha256']:raise RuntimeError('Adapter SHA mismatch')
    cfg['runtime']['teacher_adapter_checkpoint']=str(adapter)
    os.environ['BACKBONE_CHECKPOINT']=str(backbone)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,bc=build_model(cfg,backbone,device,adapter)
    model.load_state_dict(ck['state_dict'],strict=True)
    if a.disable_calibration:model.c3.set_calibration(enabled=False)
    model.set_stage(ck.get('stage','c12'))
    model.eval();teacher=model.stla.eval()
    cfg['runtime']['teacher_fingerprint']=digest_model(teacher)
    memory=FutureMemory(cache_path(cfg,'formal'))
    normal=prepare_normal_teacher_bank(cfg,memory,teacher,backbone,device)
    mean=float(ck['mean']);std=float(ck['std'])
    output=Path(a.checkpoint).resolve().parent/'evaluation'/a.split
    if a.output_tag is not None:
        output=output/'automated'/a.output_tag
    else:
        output=output/('three_switches' if any(m in THREE_SWITCH_MODES for m in a.modes) else a.controller_mode)
    output=output/(a.cf_control+'__'+a.support_control)
    if a.controller_variant:output=output/('crsi_'+a.controller_variant)
    rows=[]
    mechanism_rows=[]
    family_rows={}
    for severity in (['validation'] if a.split=='validation' else a.severities):
        eval_start=time.perf_counter()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        val=(a.split=='validation')
        source=cache_path(cfg,'formal') if val else cache_path(cfg,'evaluation',severity)
        retr=ensure_cache(cfg,source,'val' if val else 'test',severity=None if val else severity)
        rc=RetrievalCache(retr);ds=SupportSplit(source,'val' if val else 'test')
        if val and a.validation_section!='all':
            manifest=Path(a.checkpoint).resolve().parent/'validation_partitions.json'
            if not manifest.is_file():raise FileNotFoundError('Missing chronological validation partition manifest: '+str(manifest))
            partition=json.loads(manifest.read_text())
            current=chronological_partitions(ds.window_starts[:len(ds)],
                cfg['experiment']['history'],cfg['experiment']['horizon'])
            if (partition.get('window_starts_sha256')!=current.get('window_starts_sha256') or
                partition.get('boundaries')!=current.get('boundaries')):
                raise RuntimeError('Chronological validation partition does not match current dataset')
            ds_iterator=Subset(ds,partition[a.validation_section])
        else:
            ds_iterator=ds
        loader=_make_loader(ds_iterator,int(a.batch_size),False,cfg)
        # Validation replay is already built during TRAIN, including source
        # observation validity and the same Teacher SHA. Never recompute it.
        # For TEST, opt-in severity-isolated replay avoids huge default caches.
        replay=None
        if val or a.cache_test_replay:
            replay=prepare_bank(cfg,'val' if val else 'test',ds,rc,memory,
                teacher,backbone,retr,device,cache_tag=None if val else f'test_s{severity}')
        wrong_mask_changed=0
        wrong_mask_total=0
        wrong_mask_coverage_gap_sum=0.0
        if a.cf_control=='wrong_mask':
            # Deterministic real-query mask pool from the same chronological
            # subset, not synthetic spatial rolls or future labels.
            subset_indices=np.asarray(ds_iterator.indices if isinstance(ds_iterator,Subset) else range(len(ds)),dtype=np.int64)
            pool_ids=subset_indices[np.linspace(0,len(subset_indices)-1,num=min(128,len(subset_indices)),dtype=np.int64)]
            wrong_pool=torch.from_numpy(np.asarray(ds.observed[pool_ids],dtype=np.float32).copy()).to(device)
            wrong_families=(ds.meta.iloc[pool_ids].family.tolist() if 'family' in ds.meta.columns else None)
        results={m:{'prediction':[]} for m in a.modes}
        # The baseline, targets, and masks do not depend on the mechanism switch.
        shared_results={k:[] for k in ('base','truth','valid','direct')}
        diagnostics={m:MechanismDiagnostics() for m in a.modes} if a.diagnostics else {}
        p0=P0DeepDiagnostics(mean,std) if a.p0_deep else None
        print(f'V5_EVAL_START severity={severity} modes={a.modes} '
              f'max_batches={a.max_batches} output_tag={a.output_tag}',flush=True)
        evaluated_batches=0
        evaluated_indices=[]
        with torch.no_grad():
            for bi,raw in enumerate(loader,1):
                if a.max_batches and bi>a.max_batches:
                    break
                b=_batch_to_device(raw,device)
                evaluated_batches=bi
                evaluated_indices.extend(b['index'].detach().cpu().tolist())
                ev=evidence_for_batch(model,b,rc,memory,device,cfg,'M11',
                                      bank=replay,teacher=teacher,normal_bank=normal)
                if a.cf_control=='shuffled':
                    ev['residual']=ev['normal_residual']+(ev['residual']-ev['normal_residual']).roll(1,1)
                elif a.cf_control=='wrong_mask':
                    families=(ds.meta.iloc[b['index'].detach().cpu().numpy()].family.to_list()
                              if 'family' in ds.meta.columns else None)
                    wrong,changed=select_matched_wrong_mask(b['mask'],families,wrong_pool,wrong_families)
                    wrong_mask_coverage_gap_sum+=float(((b['mask'].float().mean((1,2))-wrong.float().mean((1,2))).abs()*changed.float()).sum())
                    wrong_mask_changed+=int(changed.sum())
                    wrong_mask_total+=int(changed.numel())
                    original=memory.fetch(ev['candidate_ids'],device)
                    ev['residual']=counterfactual_residuals(
                        teacher,original,wrong,int(cfg['retrieval']['replay_batch_size']),
                        'matched',strict_source_mask=True)
                with torch.autocast(device_type=device.type,dtype=torch.float16,
                     enabled=device.type=='cuda' and bool(cfg['runtime'].get('amp',True))):
                    if a.disable_shared_modes:
                        outputs={m:model(b['x'],b['mask'],b['tod'],b['dow'],mode=m,
                            controller_mode=a.controller_mode,
                            support_control=a.support_control,
                            controller_variant=a.controller_variant,
                            shuffled_negatives=(a.diagnostics and m=='M11'),**ev)
                            for m in a.modes}
                    else:
                        outputs=model.forward_modes(b['x'],b['mask'],b['tod'],b['dow'],
                            modes=a.modes,controller_mode=a.controller_mode,
                            support_control=a.support_control,
                            controller_variant=a.controller_variant,
                            shuffled_negatives=(a.diagnostics and 'M11' in a.modes),**ev)
                shared_results['base'].append((outputs[a.modes[0]]['base_prediction'].float()*std+mean).cpu().numpy())
                shared_results['truth'].append(b['y_raw'].cpu().numpy())
                shared_results['valid'].append(b['future_valid'].cpu().numpy())
                shared_results['direct'].append(b['direct_mask'].cpu().numpy())
                for m in a.modes:
                    o=outputs[m]
                    if a.diagnostics:diagnostics[m].update(o,b)
                    results[m]['prediction'].append((o['prediction'].float()*std+mean).cpu().numpy())
                if p0 is not None:
                    batch_families=(ds.meta.iloc[b['index'].detach().cpu().numpy()].family.astype(str).tolist()
                                    if 'family' in ds.meta.columns else None)
                    p0.update(outputs,b,families=batch_families,retrieval_scores=ev['candidate_scores'])
                if bi%100==0:print(f'V5_EVAL_PROGRESS severity={severity} batches={bi}',flush=True)
        if not evaluated_indices:
            raise RuntimeError('No evaluation examples available')
        # Subset metadata by true dataset indices (never use the full manifest
        # against a limited prediction array: that silently corrupts family metrics).
        evaluated_meta=ds.meta.iloc[evaluated_indices].reset_index(drop=True)
        sample_id_hash=hashlib.sha256(np.asarray(evaluated_indices,dtype='<i8').tobytes()).hexdigest()
        if p0 is not None:
            p0_folder=output/(str(severity) if val else f's{severity}')/'p0_deep'
            p0_report=p0.save(p0_folder,metadata=dict(
                checkpoint=str(Path(a.checkpoint).resolve()),
                checkpoint_sha256=file_sha256(a.checkpoint),
                split=a.split,validation_section=a.validation_section,
                severity=severity, target=cfg['experiment']['target'],
                mode_comparison=list(P0DeepDiagnostics.REQUIRED),
                calibration_disabled=bool(a.disable_calibration),
                evaluated_queries=len(evaluated_indices),
                sample_index_sha256=sample_id_hash,
                full_split_queries=len(ds), max_batches=int(a.max_batches),
                label_policy='analysis_only_never_tune_on_test'))
            print('V53_P0_DEEP_PASS '+json.dumps(dict(
                severity=severity,queries=p0_report['queries'],
                metrics=p0_report['metrics'], output=str(p0_folder))),flush=True)
        common={k:np.concatenate(v) for k,v in shared_results.items()}
        for m,d in results.items():
            d={'prediction':np.concatenate(d['prediction']),**common}
            perf=summarize_by_family(d['prediction'],d['truth'],d['valid'],d['direct'],evaluated_meta)
            base_perf=summarize_by_family(d['base'],d['truth'],d['valid'],d['direct'],evaluated_meta)
            family_rows[(str(severity),m)]=perf
            folder=output/(str(severity) if val else f's{severity}')/m;folder.mkdir(parents=True,exist_ok=True)
            save_summary(folder/'summary.json',dict(severity=severity,mode=m,
                 controller_mode=('learned' if m[-1]=='1' else 'none') if m in THREE_SWITCH_MODES else a.controller_mode,
                 controller_variant=a.controller_variant or cfg['v5'].get('c3_variant','full'),
                 calibration_disabled=bool(a.disable_calibration),
             c1_enabled=m[1]=='1',c2_enabled=m[2]=='1',
             c3_enabled=(m[-1]=='1') if m in THREE_SWITCH_MODES else (a.controller_mode=='learned'),
             cf_control=a.cf_control,
                 support_control=a.support_control,checkpoint=str(Path(a.checkpoint).resolve()),
                 checkpoint_sha256=file_sha256(a.checkpoint),
                 calibration=ck.get('calibration'),metrics=perf,base=base_perf,
                 evaluated_batches=evaluated_batches,evaluated_queries=len(evaluated_indices),
                 validation_section=a.validation_section,
                 wrong_mask_changed_queries=wrong_mask_changed,
                 wrong_mask_total_queries=wrong_mask_total,
                 wrong_mask_mean_coverage_gap=wrong_mask_coverage_gap_sum/max(wrong_mask_changed,1),
                 wrong_mask_effective_rate=(wrong_mask_changed/max(wrong_mask_total,1)) if a.cf_control=='wrong_mask' else None,
                 sample_index_sha256=sample_id_hash,
                 full_split_queries=len(ds),max_batches=a.max_batches,
                 evaluation_subset='deterministic_first_batches' if a.max_batches else 'full_split'))
            if a.diagnostics:
                diagnostic=diagnostics[m].summary()
                save_summary(folder/'diagnostics.json',diagnostic)
                pd.DataFrame(diagnostics[m].risk_coverage_rows()).to_csv(folder/'risk_coverage_curve.csv',index=False)
                mechanism_rows.append(dict(severity=severity,mode=m,
                    normal_component_gain=diagnostic.get('normal_component_gain'),
                    cf_component_gain=diagnostic.get('cf_component_gain'),
                    component_interaction_gain=diagnostic.get('component_interaction_gain'),
                    joint_harm_approx_auc=diagnostic.get('joint_harm_approx_auc'),
                    final_harm_rate=diagnostic.get('final_harm_rate')))
            if a.save_predictions:
                np.savez_compressed(folder/'predictions.npz',**d)
            row={'severity':severity,'mode':m,'evaluated_batches':evaluated_batches,
                 'evaluated_queries':len(evaluated_indices),'full_split_queries':len(ds),
                 'validation_section':a.validation_section,
                 'wrong_mask_effective_rate':wrong_mask_changed/max(wrong_mask_total,1) if a.cf_control=='wrong_mask' else None,
                 'sample_index_sha256':sample_id_hash,
                 'controller':('learned' if m[-1]=='1' else 'none') if m in THREE_SWITCH_MODES else a.controller_mode,
                 'C1':int(m[1]),'C2':int(m[2]),
                 'C3':int(m[3]) if m in THREE_SWITCH_MODES else int(a.controller_mode=='learned'),
                 **perf['ALL'],**{'base_'+k:v for k,v in base_perf['ALL'].items()}}
            rows.append(row)
            print('V5_EVAL '+json.dumps(row),flush=True)
        # Reproducible stage resource numbers; do not claim they predict Gadi
        # wall time until measured on a real V100 allocation.
        if device.type=='cuda':torch.cuda.synchronize(device)
        seconds=max(1e-9,time.perf_counter()-eval_start)
        profile=dict(split=a.split,severity=severity,queries=len(evaluated_indices),
                     elapsed_seconds=round(seconds,3),
                     queries_per_second=round(len(evaluated_indices)/seconds,2),
                     peak_cpu_rss_gib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,3),
                     replay_from_bank=(replay is not None),
                     shared_forward=(not a.disable_shared_modes),
                     mode_count=len(a.modes))
        if device.type=='cuda':
            profile['peak_gpu_allocated_gib']=round(torch.cuda.max_memory_allocated(device)/1024**3,3)
            profile['peak_gpu_reserved_gib']=round(torch.cuda.max_memory_reserved(device)/1024**3,3)
        output.mkdir(exist_ok=True,parents=True)
        (output/f'resource_profile_{severity}.json').write_text(json.dumps(profile,indent=2))
        print('V5_RESOURCE_PROFILE '+json.dumps(profile),flush=True)
    if a.cf_control=='wrong_mask' and wrong_mask_changed==0:
        raise RuntimeError('No distinct wrong-mask controls were available; comparison invalid')
    summary=pd.DataFrame(rows)
    output.mkdir(exist_ok=True,parents=True)
    summary.to_csv(output/'mechanism_v5.csv',index=False)
    if mechanism_rows:
        pd.DataFrame(mechanism_rows).to_csv(output/'component_interaction_by_severity.csv',index=False)
    c3_family_rows=[]
    for key,performance in family_rows.items():
        severity,mode=key
        if mode!='M110' or (severity,'M111') not in family_rows:continue
        other=family_rows[(severity,'M111')]
        for family in sorted(set(performance)&set(other)-{'ALL'}):
            c3_family_rows.append(dict(severity=severity,fault_family=family,
                c3_off_MAE=performance[family]['MAE'],
                c3_on_MAE=other[family]['MAE'],
                c3_gain_MAE=performance[family]['MAE']-other[family]['MAE']))
    if c3_family_rows:
        pd.DataFrame(c3_family_rows).to_csv(output/'c3_gain_by_fault_family.csv',index=False)
    if all(m in a.modes for m in THREE_SWITCH_MODES):
        matrix=summary.pivot(index='severity',columns='mode',values='MAE')
        gains={'severity':matrix.index}
        for cf in (0,1):
            for support in (0,1):
                gains[f'C3_gain_CF{cf}_Support{support}']=(
                    matrix[f'M{cf}{support}0']-matrix[f'M{cf}{support}1'])
        for support in (0,1):
            for c3 in (0,1):
                gains[f'C1_gain_Support{support}_C3{c3}']=(
                    matrix[f'M0{support}{c3}']-matrix[f'M1{support}{c3}'])
        for cf in (0,1):
            for c3 in (0,1):
                gains[f'C2_gain_CF{cf}_C3{c3}']=(
                    matrix[f'M{cf}0{c3}']-matrix[f'M{cf}1{c3}'])
        pd.DataFrame(gains).to_csv(output/'mechanism_gains_v5_3switch.csv',index=False)
    if all(m in a.modes for m in MODES):
        matrix=summary.pivot(index='severity',columns='mode',values='MAE')
        gain=pd.DataFrame({'severity':matrix.index,
            'CF_without_support':matrix['M00']-matrix['M10'],
            'CF_with_support':matrix['M01']-matrix['M11'],
            'Support_without_CF':matrix['M00']-matrix['M01'],
            'Support_with_CF':matrix['M10']-matrix['M11'],
            'CF_Support_interaction':matrix['M10']+matrix['M01']-matrix['M00']-matrix['M11']})
        gain.to_csv(output/'mechanism_gains_v5_2switch.csv',index=False)
    print(f'V5_EVALUATION_PASS output={output}',flush=True)
if __name__=='__main__':main()
