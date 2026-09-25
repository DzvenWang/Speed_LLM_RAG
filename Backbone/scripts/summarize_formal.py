#!/usr/bin/env python3
from pathlib import Path
import json, sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from backbone.utils import load_yaml, write_json, sha256_file
from backbone.identity import load_cache_identity, require_identity, require_cache_version_compatibility
from backbone.shared_cache import cache_path

def main():
    cfg=load_yaml(ROOT/'configs/backbone.yaml'); exp=Path(cfg['project']['exp_root']); sg=cfg['experiment']['subgraph']; expected=load_cache_identity(cache_path(cfg,'formal')); require_cache_version_compatibility(cfg,expected,'formal')
    seeds=[int(x) for x in cfg['experiment']['formal_seeds']]; root=exp/'PatchSTG'/sg/'formal'; files=sorted(root.glob('*/summary.json')) if root.exists() else []; by_seed={}
    for f in files:
        try:
            s=json.load(open(f)); require_identity(s,expected,str(f)); seed=int(s['seed']);
            if str(s.get('project_version'))!=str(cfg['version']): continue
        except Exception: continue
        if seed in seeds and (seed not in by_seed or f.stat().st_mtime>by_seed[seed][0].stat().st_mtime): by_seed[seed]=(f,s)
    missing=[x for x in seeds if x not in by_seed]
    if missing: raise SystemExit(f'PatchSTG missing current-contract formal seeds {missing}')
    rows=[]; checkpoints={}
    for seed in seeds:
        f,s=by_seed[seed]; ck=f.parent/'best.pt'
        if not ck.exists(): raise SystemExit(f'Missing checkpoint {ck}')
        v=s['validation']; sha=sha256_file(ck); checkpoints[str(seed)]={'checkpoint':str(ck),'checkpoint_sha256':sha,'run_dir':str(f.parent),'best_epoch':int(s['best_epoch']),'val_MAE':float(v['MAE'])}
        rows.append({'model':'PatchSTG','seed':seed,'run_dir':str(f.parent),'checkpoint':str(ck),'checkpoint_sha256':sha,'parameters':int(s['parameters']),'batch_size':int(s['batch_size']),'best_epoch':int(s['best_epoch']),'protocol_name':expected['protocol_name'],'protocol_version':expected['protocol_version'],'severity':expected['severity'],'protocol_stage_hash':expected['protocol_stage_hash'],'cache_fingerprint':expected['cache_fingerprint'],'val_MAE':v['MAE'],'val_RMSE':v['RMSE'],'val_masked_MAE':v['directly_masked_MAE'],'val_masked_RMSE':v.get('directly_masked_RMSE'),'val_support_MAE':v['support_dependent_MAE'],'val_support_RMSE':v.get('support_dependent_RMSE'),'val_H12_MAE':v['H12_MAE'],'val_H12_RMSE':v.get('H12_RMSE'),'val_H12_support_MAE':v['H12_support_dependent_MAE'],'val_H12_support_RMSE':v.get('H12_support_dependent_RMSE'),'val_inference_ms_per_query':v['inference_ms_per_query']})
    runs=pd.DataFrame(rows); runs.to_csv(exp/'backbone_formal_runs.csv',index=False)
    numeric=[c for c in runs.columns if c.startswith('val_') and c not in ('val_inference_ms_per_query',)] + ['val_inference_ms_per_query']; agg={'model':'PatchSTG','n_seeds':len(runs)}
    for c in numeric:
        vals=pd.to_numeric(runs[c],errors='coerce'); agg['mean_'+c]=float(vals.mean()); agg['std_'+c]=float(vals.std(ddof=0))
    pd.DataFrame([agg]).to_csv(exp/'backbone_formal_summary.csv',index=False)
    ref_seed=int(cfg['experiment']['reference_seed']); ref=checkpoints[str(ref_seed)]
    frozen={'schema_version':4,'project':'Backbone','project_version':cfg['version'],'selected_model':'PatchSTG','selection_mode':'configured_patchstg_three_seed_formal_stability','selection_uses_test':False,'run_identity':expected,'formal_seeds':seeds,'formal_checkpoints':checkpoints,'reference_seed':ref_seed,'reference_checkpoint':ref['checkpoint'],'reference_checkpoint_sha256':ref['checkpoint_sha256'],'reference_run_dir':ref['run_dir'],'formal_mean_validation_MAE':agg['mean_val_MAE'],'formal_std_validation_MAE':agg['std_val_MAE'],'evaluation_severities':[int(x) for x in cfg['experiment']['evaluation_severities']]}
    write_json(exp/'FROZEN_BACKBONE.json',frozen); print(runs.to_string(index=False)); print('\nFORMAL SUMMARY'); print(pd.DataFrame([agg]).to_string(index=False)); print('\nFROZEN_BACKBONE',exp/'FROZEN_BACKBONE.json')
if __name__=='__main__': main()
