#!/usr/bin/env python3
"""Summarize the configured downstream reference backbone over formal seeds.

Backbone v0.5.0 treats five-model screening as a comparison/verification table,
while the downstream numerical reference is explicitly configured (PatchSTG).
This removes the unnecessary dependency between formal reference training and a
screening shortlist file.
"""
from pathlib import Path
import json, sys
import pandas as pd
GADI_ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone')
ROOT=GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backbone.utils import load_yaml,write_json,sha256_file
from backbone.identity import load_cache_identity,require_identity,require_cache_version_compatibility
from backbone.shared_cache import cache_path


def main():
    cfg=load_yaml(ROOT/'configs/backbone_selection.yaml'); exp=Path(cfg['project']['exp_root']); sg=cfg['experiment']['subgraph']
    expected=load_cache_identity(cache_path(cfg,'formal')); require_cache_version_compatibility(cfg,expected,'formal')
    model=str(cfg['selection'].get('reference_model','PatchSTG')); seeds=[int(x) for x in cfg['experiment']['formal']['seeds']]
    root=exp/model/sg/'formal'; files=sorted(root.glob('*/summary.json')) if root.exists() else []
    by_seed={}
    for f in files:
        try:
            s=json.load(open(f)); require_identity(s,expected,str(f));
            if str(s.get('project_version')) != str(cfg['version']):
                raise ValueError(f'code project version {s.get("project_version")} != {cfg["version"]}')
            seed=int(s['seed'])
        except Exception:
            continue
        if seed in seeds and (seed not in by_seed or f.stat().st_mtime>by_seed[seed][0].stat().st_mtime): by_seed[seed]=(f,s)
    missing=[s for s in seeds if s not in by_seed]
    if missing: raise SystemExit(f'{model}: missing current-contract formal seeds {missing}')
    rows=[]
    for seed in seeds:
        f,s=by_seed[seed]; ck=f.parent/'best.pt'
        if not ck.exists(): raise SystemExit(f'Missing checkpoint {ck}')
        v=s['validation']
        rows.append({'model':model,'seed':seed,'run_dir':str(f.parent),'checkpoint':str(ck),'checkpoint_sha256':sha256_file(ck),'parameters':int(s['parameters']),'batch_size':int(s['batch_size']),'best_epoch':int(s['best_epoch']),'code_project_version':s.get('project_version'),'shared_contract_version':expected['shared_contract_version'],'data_contract_revision':expected['data_contract_revision'],'time_feature_revision':expected['time_feature_revision'],'metrics_revision':expected['metrics_revision'],'prediction_contract_revision':expected['prediction_contract_revision'],'time_feature_fingerprint':expected['time_feature_fingerprint'],'protocol_stage_hash':expected['protocol_stage_hash'],'cache_fingerprint':expected['cache_fingerprint'],'val_MAE':v['MAE'],'val_RMSE':v['RMSE'],'val_masked_MAE':v['directly_masked_MAE'],'val_masked_RMSE':v.get('directly_masked_RMSE'),'val_support_MAE':v['support_dependent_MAE'],'val_support_RMSE':v.get('support_dependent_RMSE'),'val_H12_MAE':v['H12_MAE'],'val_H12_RMSE':v.get('H12_RMSE'),'val_H12_support_MAE':v['H12_support_dependent_MAE'],'val_H12_support_RMSE':v.get('H12_support_dependent_RMSE'),'val_inference_ms_per_query':v['inference_ms_per_query']})
    runs=pd.DataFrame(rows); runs.to_csv(exp/'backbone_formal_runs.csv',index=False)
    numeric=['val_MAE','val_RMSE','val_masked_MAE','val_masked_RMSE','val_support_MAE','val_support_RMSE','val_H12_MAE','val_H12_RMSE','val_H12_support_MAE','val_H12_support_RMSE','val_inference_ms_per_query']
    agg={'model':model,'n_seeds':len(runs)}
    for c in numeric:
        vals=pd.to_numeric(runs[c],errors='coerce'); agg['mean_'+c]=float(vals.mean()); agg['std_'+c]=float(vals.std(ddof=0))
    pd.DataFrame([agg]).to_csv(exp/'backbone_formal_summary.csv',index=False)
    ref_seed=int(cfg['experiment']['formal']['reference_seed']); ref=runs.loc[runs.seed.eq(ref_seed)]
    if len(ref)!=1: raise SystemExit(f'Need exactly one reference seed {ref_seed}; got {len(ref)}')
    ref=ref.iloc[0]
    frozen={'schema_version':3,'project':'Backbone','project_version':cfg['version'],'shared_contract_version':expected['shared_contract_version'],'selected_model':model,'selection_mode':'configured_reference_model_with_three_seed_formal_stability','selection_uses_test':False,'run_identity':expected,'formal_seeds':seeds,'reference_seed':ref_seed,'reference_checkpoint':str(ref.checkpoint),'reference_checkpoint_sha256':str(ref.checkpoint_sha256),'reference_run_dir':str(ref.run_dir),'formal_mean_validation_MAE':agg['mean_val_MAE'],'formal_std_validation_MAE':agg['std_val_MAE'],'screening_summary_optional':str(exp/'backbone_screening_summary.csv'),'aggregate_rows':[agg]}
    write_json(exp/'FROZEN_BACKBONE.json',frozen)
    print(runs.to_string(index=False)); print('\nFORMAL SUMMARY'); print(pd.DataFrame([agg]).to_string(index=False)); print('\nFROZEN BACKBONE:',model); print('REFERENCE CHECKPOINT:',ref.checkpoint); print('REFERENCE CHECKPOINT SHA256:',ref.checkpoint_sha256)
if __name__=='__main__': main()
