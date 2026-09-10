#!/usr/bin/env python3
from pathlib import Path
import json, sys
import pandas as pd
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone'); sys.path.insert(0,str(ROOT))
from backbone.utils import load_yaml, write_json
from backbone.selection import formal_aggregate, choose_formal
from backbone.identity import load_cache_identity, require_identity, require_cache_version_compatibility
cfg=load_yaml(ROOT/'configs/backbone_selection.yaml'); exp=Path(cfg['project']['exp_root']); sg=cfg['experiment']['subgraph']
expected=load_cache_identity(exp/'_cache'/sg/'formal')
require_cache_version_compatibility(cfg, expected, 'formal')
short_path=exp/'screening_shortlist.json'
if not short_path.exists(): raise SystemExit('Missing screening_shortlist.json')
short_obj=json.load(open(short_path))
# Shortlist may come from development Protocol, but must be from current project version and carry tracked identity.
if str(short_obj.get('project_version'))!=str(cfg.get('version')) or not isinstance(short_obj.get('run_identity'),dict):
    raise SystemExit('screening_shortlist.json is old/untracked; rerun current screening summary.')
short=short_obj['shortlist']; expected_seeds=[int(x) for x in cfg['experiment']['formal']['seeds']]
rows=[]
for model in short:
    root=exp/model/sg/'formal'; files=sorted(root.glob('*/summary.json')) if root.exists() else []
    by_seed={}
    for f in files:
        try:
            s=json.load(open(f)); require_identity(s,expected,str(f)); seed=int(s['seed'])
        except Exception:
            continue
        if seed in expected_seeds and (seed not in by_seed or f.stat().st_mtime>by_seed[seed][0].stat().st_mtime): by_seed[seed]=(f,s)
    for seed in expected_seeds:
        if seed not in by_seed: continue
        f,s=by_seed[seed]; v=s['validation']
        rows.append({'model':model,'seed':seed,'run_dir':str(f.parent),'checkpoint':str(f.parent/'best.pt'),'parameters':int(s['parameters']),
          'batch_size':int(s['batch_size']),'project_version':s.get('project_version'),'protocol_stage_hash':expected['protocol_stage_hash'],
          'cache_fingerprint':expected['cache_fingerprint'],'train_view_hash':expected['split_view_hashes']['train'],'val_view_hash':expected['split_view_hashes']['val'],
          'val_MAE':v['MAE'],'val_RMSE':v['RMSE'],'val_masked_MAE':v['directly_masked_MAE'],'val_masked_RMSE':v.get('directly_masked_RMSE'),'val_support_MAE':v['support_dependent_MAE'],'val_support_RMSE':v.get('support_dependent_RMSE'),
          'val_H12_MAE':v['H12_MAE'],'val_H12_RMSE':v.get('H12_RMSE'),'val_H12_support_MAE':v['H12_support_dependent_MAE'],'val_H12_support_RMSE':v.get('H12_support_dependent_RMSE'),'val_inference_ms_per_query':v['inference_ms_per_query']})
runs=pd.DataFrame(rows)
for model in short:
    got=sorted(runs.loc[runs.model.eq(model),'seed'].unique().tolist()) if len(runs) else []
    if got!=sorted(expected_seeds): raise SystemExit(f'{model}: expected current-identity seeds {expected_seeds}, got {got}')
for col in ['project_version','protocol_stage_hash','cache_fingerprint','train_view_hash','val_view_hash']:
    if runs[col].nunique()!=1: raise SystemExit(f'FAIL mixed formal identity in {col}: {runs[col].unique().tolist()}')
runs.to_csv(exp/'backbone_formal_runs.csv',index=False)
agg=formal_aggregate(runs); sel=cfg['selection']; agg,selected,best=choose_formal(agg,float(sel['overall_tolerance']),float(sel['support_tie_tolerance_rel'])); agg.to_csv(exp/'backbone_formal_summary.csv',index=False)
ref_seed=int(cfg['experiment']['formal']['reference_seed']); ref=runs[(runs.model.eq(selected))&(runs.seed.eq(ref_seed))]
if len(ref)!=1: raise SystemExit(f'Need exactly one seed-{ref_seed} formal run for selected {selected}; got {len(ref)}')
ref=ref.iloc[0]
frozen={'project_version':cfg.get('version'),'selected_model':selected,'selected_architecture_by':'Validation-only 3-seed formal protocol','selection_uses_test':False,
 'run_identity':expected,'formal_seeds':expected_seeds,'reference_seed':ref_seed,'reference_checkpoint':str(ref.checkpoint),'reference_run_dir':str(ref.run_dir),
 'best_mean_validation_MAE':best,'rule':'First keep architectures whose 3-seed mean Val Overall MAE is within 2% of the best. Then choose the lowest mean Val support-dependent MAE. If support MAE is within 0.5%, compare masked MAE; if masked MAE is also within 0.5%, prefer lower Val MAE seed variance, then inference time, then parameters.',
 'aggregate_rows':agg.to_dict(orient='records')}
write_json(exp/'FROZEN_BACKBONE.json',frozen)
print(runs.sort_values(['model','seed']).to_string(index=False)); print('\nAGGREGATE'); print(agg.to_string(index=False)); print('\nFROZEN BACKBONE:',selected); print('PROTOCOL_STAGE_HASH='+expected['protocol_stage_hash']); print('REFERENCE CHECKPOINT:',ref.checkpoint)
