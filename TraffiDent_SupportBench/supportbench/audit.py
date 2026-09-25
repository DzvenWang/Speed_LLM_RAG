from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import SupportBench
from .sensing import materialize_mask
from .dependency import dependency_context_loss
from .utils import read_json


def _layout_sets(path: Path) -> tuple[set[tuple[int,...]], set[tuple[int,...]]]:
    obj=read_json(path,{}) or {}; ids=set(); oods=set()
    for _,part in (obj.get('severities') or {}).items():
        for item in part.get('id',[]): ids.add(tuple(sorted(int(x) for x in item['nodes'])))
        for item in part.get('ood',[]): oods.add(tuple(sorted(int(x) for x in item['nodes'])))
    return ids,oods



def _layout_min_for_severity(token: str, key: str, default: int, overrides: dict | None = None) -> int:
    block=(overrides or {}).get(str(token),{})
    value=block.get(str(key),default) if isinstance(block,dict) else default
    return max(1,int(value))


def audit_subgraph(sg_dir: str | Path, config: str | Path | None = None, max_per_stratum: int = 3) -> dict:
    sg_dir=Path(sg_dir); bench=SupportBench(sg_dir.name,root=sg_dir.parent,config=config)
    report={'subgraph':sg_dir.name,'checks':{},'stratified_mask_audit':[],'errors':[],'warnings':[]}

    try:
        splits=bench.split_manifest['splits']
        assert splits['train']['end_exclusive'] <= splits['val']['start'] <= splits['val']['end_exclusive'] <= splits['test']['start']
        report['checks']['split_order']='PASS'
    except Exception as e:
        report['checks']['split_order']='FAIL'; report['errors'].append(f'split_order: {e}')

    # Predictive-dependency statistics must be fitted on Train only, while
    # lag duration remains at the raw traffic resolution (5 min by default).
    try:
        dep=np.load(bench.sg_dir/'graph'/'predictive_dependency.npz')
        tr=bench.split_range('train')
        dep_ok=int(dep['train_start'])==int(tr['start']) and int(dep['train_end'])==int(tr['end_exclusive'])
        report['checks']['predictive_dependency_train_only']='PASS' if dep_ok else 'FAIL'
        if not dep_ok: report['errors'].append('Predictive dependency interval is not exactly the Train split')

        freq=int(bench.cfg['forecast']['freq_minutes'])
        lag_step=int(dep['lag_step_minutes']) if 'lag_step_minutes' in dep.files else -1
        stride_steps=int(dep['sample_stride_steps']) if 'sample_stride_steps' in dep.files else int(dep['sample_stride'])
        stride_minutes=int(dep['sample_stride_minutes']) if 'sample_stride_minutes' in dep.files else -1
        lag_semantics_ok=(lag_step==freq and stride_minutes==stride_steps*freq)
        report['checks']['predictive_dependency_raw_lag_semantics']='PASS' if lag_semantics_ok else 'FAIL'
        report['checks']['predictive_dependency_lag_step_minutes']=lag_step
        report['checks']['predictive_dependency_sample_stride_minutes']=stride_minutes
        if not lag_semantics_ok:
            report['errors'].append(
                f'Predictive dependency lag semantics invalid: lag_step_minutes={lag_step}, '
                f'freq_minutes={freq}, sample_stride_steps={stride_steps}, sample_stride_minutes={stride_minutes}'
            )
    except Exception as e:
        report['checks']['predictive_dependency_train_only']='FAIL'
        report['checks']['predictive_dependency_raw_lag_semantics']='FAIL'
        report['errors'].append(f'predictive_dependency: {e}')

    for split in ['train','val','test']:
        starts=bench.window_starts(split); r=bench.split_range(split)
        ok=np.all((starts>=int(r['start']))&(starts+bench.H+bench.P<=int(r['end_exclusive']))) if len(starts) else True
        report['checks'][f'{split}_windows_inside_split']='PASS' if ok else 'FAIL'
        if not ok: report['errors'].append(f'{split}: window leakage across temporal split')
        dev=bench.window_starts(split,'development')
        dev_ok=set(map(int,dev)).issubset(set(map(int,starts))) and len(dev)<=int(bench.cfg['budgets'][f'development_{split}_windows'])
        report['checks'][f'{split}_development_subset']='PASS' if dev_ok else 'FAIL'
        if not dev_ok: report['errors'].append(f'{split}: invalid development subset')

    complete_ok=True
    for split in ['train','val','test']:
        starts=bench.window_starts(split)
        pick=starts[np.linspace(0,max(len(starts)-1,0),num=min(1000,len(starts)),dtype=int)] if len(starts) else []
        for s in pick:
            block=np.asarray(bench.valid[int(s):int(s)+bench.H+bench.P,:,bench.speed_idx],dtype=bool)
            if float(block.mean())+1e-12 < float(bench.manifest['controlled_complete_min_fraction']): complete_ok=False; break
    report['checks']['controlled_window_validity']='PASS' if complete_ok else 'FAIL'
    if not complete_ok: report['errors'].append('Controlled window validity below configured threshold')

    # Stratified mask audit. This replaces the old random sample so every
    # family/severity/distribution protocol is guaranteed to be inspected.
    tol=float(bench.cfg['sensing']['effective_ratio_tolerance']); deterministic=True; worst=0.0; manifest_worst=0.0
    group_cols=['family','severity','distribution']
    for keys,g in bench.views.groupby(group_cols,dropna=False,sort=True):
        family,severity,distribution=keys; family=str(family); severity=float(severity); distribution=str(distribution)
        if len(g)>max_per_stratum:
            pos=np.linspace(0,len(g)-1,num=max_per_stratum,dtype=int); sample_rows=g.iloc[pos]
        else: sample_rows=g
        actual=[]; declared=[]
        for _,row in sample_rows.iterrows():
            spec=row.to_dict(); s=int(spec['window_start']); valid=np.asarray(bench.valid[s:s+bench.H,:,bench.speed_idx],dtype=bool)
            a=materialize_mask(spec,bench.H,bench.N,bench.sg_dir,valid); b=materialize_mask(spec,bench.H,bench.N,bench.sg_dir,valid)
            if not np.array_equal(a,b): deterministic=False
            actual.append(float(a.mean())); declared.append(float(spec.get('effective_missing_ratio',np.nan)))
        arr=np.asarray(actual,float); dec=np.asarray(declared,float)
        ratio_err=float(np.max(np.abs(arr-severity))) if family!='complete' and len(arr) else 0.0
        manifest_err=float(np.nanmax(np.abs(arr-dec))) if len(arr) and np.isfinite(dec).any() else 0.0
        worst=max(worst,ratio_err); manifest_worst=max(manifest_worst,manifest_err)
        status='PASS' if ratio_err<=tol+1e-12 and manifest_err<=1e-9 else 'FAIL'
        report['stratified_mask_audit'].append({
            'family':family,'severity':severity,'distribution':distribution,'n_views':int(len(g)),'n_checked':int(len(sample_rows)),
            'actual_mean':float(arr.mean()) if len(arr) else 0.0,'actual_min':float(arr.min()) if len(arr) else 0.0,'actual_max':float(arr.max()) if len(arr) else 0.0,
            'max_abs_error_to_nominal':ratio_err,'max_abs_error_to_manifest':manifest_err,'status':status,
        })
        if status=='FAIL':
            report['errors'].append(f'Mask protocol {family}/{severity:.2f}/{distribution}: ratio_err={ratio_err:.4f}, manifest_err={manifest_err:.4g}')
    report['checks']['view_determinism']='PASS' if deterministic else 'FAIL'
    if not deterministic: report['errors'].append('Mask materialisation is not deterministic')
    report['checks']['mask_ratio_max_abs_error']=worst
    report['checks']['manifest_ratio_max_abs_error']=manifest_worst

    # Fixed OOD layouts must be genuinely unseen node configurations while
    # preserving matched severity. OOD is required only for configured OOD
    # severities (main severity by default); high-severity OOD catalogues that
    # are not part of any protocol are deliberately not generated.
    ood_tokens={f"s{int(round(100*float(x))):02d}" for x in bench.cfg['sensing'].get('ood_severities',[bench.cfg['sensing']['main_severity']])}
    min_overrides=bench.cfg['sensing'].get('layout_min_overrides',{})
    layout_specs=[
        ('regional','regional_layouts.json','regional_id','regional_ood',min(int(bench.cfg['sensing'].get('regional_min_id_layouts_per_severity',1)),int(bench.cfg['sensing'].get('regional_id_layouts_per_severity',1))),min(int(bench.cfg['sensing'].get('regional_min_ood_layouts_per_severity',1)),int(bench.cfg['sensing'].get('regional_ood_layouts_per_severity',1))),ood_tokens),
        ('directional','directional_layouts.json','directional_id','directional_ood',min(int(bench.cfg['sensing'].get('directional_min_id_layouts_per_severity',1)),int(bench.cfg['sensing'].get('directional_id_layouts_per_severity',1))),min(int(bench.cfg['sensing'].get('directional_min_ood_layouts_per_severity',1)),int(bench.cfg['sensing'].get('directional_ood_layouts_per_severity',1))),ood_tokens),
        ('predictive_regional','predictive_regional_layouts.json','predictive_regional_id','predictive_regional_ood',min(int(bench.cfg['sensing'].get('predictive_regional_min_id_layouts_per_severity',1)),int(bench.cfg['sensing'].get('predictive_regional_id_layouts_per_severity',1))),min(int(bench.cfg['sensing'].get('predictive_regional_min_ood_layouts_per_severity',1)),int(bench.cfg['sensing'].get('predictive_regional_ood_layouts_per_severity',1))),ood_tokens),
        ('predictive_directional','predictive_directional_layouts.json','predictive_directional_id','predictive_directional_ood',min(int(bench.cfg['sensing'].get('predictive_directional_min_id_layouts_per_severity',1)),int(bench.cfg['sensing'].get('predictive_directional_id_layouts_per_severity',1))),min(int(bench.cfg['sensing'].get('predictive_directional_min_ood_layouts_per_severity',1)),int(bench.cfg['sensing'].get('predictive_directional_ood_layouts_per_severity',1))),ood_tokens),
    ]
    for fam,file_name,min_id_key,min_ood_key,default_min_id,default_min_ood,required_ood_tokens in layout_specs:
        obj=read_json(bench.sg_dir/'graph'/'layouts'/file_name,{}) or {}
        all_ids=set(); all_oods=set(); fam_ok=True
        for token,part in sorted((obj.get('severities') or {}).items()):
            min_id=_layout_min_for_severity(token,min_id_key,default_min_id,min_overrides)
            min_ood=_layout_min_for_severity(token,min_ood_key,default_min_ood,min_overrides)
            ids={tuple(sorted(int(x) for x in item['nodes'])) for item in part.get('id',[])}
            oods={tuple(sorted(int(x) for x in item['nodes'])) for item in part.get('ood',[])}
            all_ids |= ids; all_oods |= oods
            if len(ids) < min_id:
                fam_ok=False; report['errors'].append(f'{fam}/{token}: only {len(ids)} ID layouts; minimum={min_id}')
            if token in required_ood_tokens:
                if len(oods) < min_ood:
                    fam_ok=False; report['errors'].append(f'{fam}/{token}: only {len(oods)} OOD layouts; minimum={min_ood}')
                overlap=ids & oods
                if overlap:
                    fam_ok=False; report['errors'].append(f'{fam}/{token}: ID/OOD fixed layout overlap={len(overlap)}')
            elif len(oods) != 0:
                fam_ok=False; report['errors'].append(f'{fam}/{token}: unexpected OOD layouts outside configured ood_severities')
            report['checks'][f'{fam}_{token}_id_layout_count']=len(ids)
            report['checks'][f'{fam}_{token}_ood_layout_count']=len(oods)
        report['checks'][f'{fam}_id_ood_layout_disjoint']='PASS' if fam_ok else 'FAIL'
        report['checks'][f'{fam}_id_layout_count']=len(all_ids)
        report['checks'][f'{fam}_ood_layout_count']=len(all_oods)

    # Regional masks should remain spatially concentrated. On disconnected road
    # graphs this is a compactness diagnostic rather than a correctness failure.
    regional=bench.views[bench.views.family.isin(['regional_outage','predictive_regional_outage'])]
    if len(regional) and 'regional_largest_component_fraction' in regional.columns:
        min_frac=float(pd.to_numeric(regional['regional_largest_component_fraction'],errors='coerce').min())
        threshold=float(bench.cfg['sensing']['regional_min_largest_component_fraction'])
        report['checks']['regional_min_largest_component_fraction']=min_frac
        if min_frac+1e-12 < threshold:
            report['warnings'].append(f'Regional compactness {min_frac:.3f} below configured diagnostic threshold {threshold:.3f}; graph may be highly disconnected.')

    # Predictive structured masks are intentionally support-oriented.  On
    # matched Validation windows, they should remove at least as much Train-defined
    # predictive-neighbour context as the corresponding general structured mask.
    # This is a diagnostic, not a Test-label filter: it uses only the fixed mask
    # and the Train-only dependency graph. Validation is used here so this
    # diagnostic can guide benchmark tuning without consulting Test queries.
    report['predictive_context_audit']=[]
    dep_cfg=bench.cfg.get('dependency',{})
    diag_n=max(1,int(dep_cfg.get('predictive_context_diagnostic_views',64)))
    min_ratio=float(dep_cfg.get('predictive_context_min_ratio',1.05))
    strict_context=bool(dep_cfg.get('predictive_context_strict',False))
    main=float(bench.cfg['sensing']['main_severity'])
    for general_fam,predictive_fam in [
        ('regional_outage','predictive_regional_outage'),
        ('directional_outage','predictive_directional_outage'),
    ]:
        gd=bench.views[(bench.views.split=='val')&(bench.views.family==general_fam)&(bench.views.distribution=='id')&np.isclose(bench.views.severity.astype(float),main)&(bench.views.instance.astype(int)==0)].copy()
        pdv=bench.views[(bench.views.split=='val')&(bench.views.family==predictive_fam)&(bench.views.distribution=='id')&np.isclose(bench.views.severity.astype(float),main)&(bench.views.instance.astype(int)==0)].copy()
        gmap={int(r.window_start):r for _,r in gd.iterrows()}
        pmap={int(r.window_start):r for _,r in pdv.iterrows()}
        common=sorted(set(gmap)&set(pmap))
        if len(common)>diag_n:
            pos=np.linspace(0,len(common)-1,num=diag_n,dtype=int); common=[common[int(i)] for i in pos]
        general_losses=[]; predictive_losses=[]
        for ws in common:
            for row,out in [(gmap[ws],general_losses),(pmap[ws],predictive_losses)]:
                spec=row.to_dict(); s0=int(spec['window_start'])
                valid=np.asarray(bench.valid[s0:s0+bench.H,:,bench.speed_idx],dtype=bool)
                mask=materialize_mask(spec,bench.H,bench.N,bench.sg_dir,valid)
                per_target=dependency_context_loss(mask,bench.dependency_ids,bench.dependency_weights)
                out.append(float(np.mean(per_target)))
        gmean=float(np.mean(general_losses)) if general_losses else float('nan')
        pmean=float(np.mean(predictive_losses)) if predictive_losses else float('nan')
        ratio=float(pmean/max(gmean,1e-12)) if np.isfinite(gmean) and np.isfinite(pmean) else float('nan')
        status='PASS' if np.isfinite(ratio) and ratio+1e-12>=min_ratio else 'LOW_GAIN'
        report['predictive_context_audit'].append({
            'general_family':general_fam,'predictive_family':predictive_fam,'severity':main,'split':'val',
            'n_matched_views':int(len(common)),'general_context_loss_mean':gmean,
            'predictive_context_loss_mean':pmean,'predictive_to_general_ratio':ratio,
            'configured_min_ratio':min_ratio,'status':status,
        })
        report['checks'][f'{predictive_fam}_context_loss_ratio']=ratio
        if status!='PASS':
            msg=(f'{predictive_fam}: predictive/general context-loss ratio={ratio:.3f} '
                 f'< diagnostic target {min_ratio:.3f}; predictive layouts may not be sufficiently support-oriented.')
            if strict_context: report['errors'].append(msg)
            else: report['warnings'].append(msg)

    # Paired main views must refer to exactly the same base window/group.
    main=float(bench.cfg['sensing']['main_severity']); paired_ok=True
    for split in ['val','test']:
        d=bench.views[(bench.views.split==split)&(bench.views.distribution=='id')&np.isclose(bench.views.severity.astype(float),main)]
        for (w,fam),g in d.groupby(['window_start','family']):
            if fam=='complete': continue
            if len(g)>=2 and len(set(g.view_group_id.astype(str)))!=1: paired_ok=False; break
    report['checks']['paired_views_share_base_window']='PASS' if paired_ok else 'FAIL'
    if not paired_ok: report['errors'].append('Paired sensing views do not share a base view_group_id')

    tr=bench.split_range('train'); ms=bench.memory_starts
    memory_ok=np.all((ms>=int(tr['start']))&(ms+bench.H+bench.P<=int(tr['end_exclusive']))) if len(ms) else False
    report['checks']['memory_train_only']='PASS' if memory_ok else 'FAIL'
    if not memory_ok: report['errors'].append('Historical memory contains non-Train episodes')
    dev_ids=bench.memory_episode_ids_for_budget('development'); dev_ok=set(map(int,dev_ids)).issubset(set(range(len(ms))))
    report['checks']['development_memory_subset']='PASS' if dev_ok else 'FAIL'
    if not dev_ok: report['errors'].append('Development memory is not a subset of full Train memory')
    stride=int(bench.cfg['memory']['episode_stride']); spacing_ok=bool(len(ms)<2 or np.all(np.diff(ms)>=stride))
    report['checks']['memory_stride']='PASS' if spacing_ok else 'FAIL'
    if not spacing_ok: report['errors'].append('Memory episodes violate configured minimum stride')

    patches_ok=bench.patch_ids.shape[0]==bench.N and np.all(bench.patch_ids[:,0]==np.arange(bench.N))
    report['checks']['local_patch_target_first']='PASS' if patches_ok else 'FAIL'
    if not patches_ok: report['errors'].append('Local patches do not keep target node first')

    required_protocols=['track_general_development.csv','track_main_support_development.csv','track_predictive_structured_development.csv','challenge_val_development.csv','challenge_test_development.csv','gap2_val_development.csv','gap3_test_development.csv']
    missing_protocols=[x for x in required_protocols if not (bench.sg_dir/'protocols'/'resolved'/x).exists()]
    report['checks']['resolved_protocols']='PASS' if not missing_protocols else 'FAIL'
    if missing_protocols: report['errors'].append(f'Missing resolved protocols: {missing_protocols}')

    report['status']='PASS' if not report['errors'] else 'FAIL'
    return report
