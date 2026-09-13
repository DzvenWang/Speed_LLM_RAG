#!/usr/bin/env python3
from pathlib import Path
import json, sys
import pandas as pd
ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone')
sys.path.insert(0, str(ROOT))
from backbone.utils import load_yaml, write_json
from backbone.selection import MODELS, screening_shortlist
from backbone.identity import load_cache_identity, require_identity, require_cache_version_compatibility
from backbone.shared_cache import cache_path

cfg = load_yaml(ROOT / 'configs/backbone_selection.yaml')
exp = Path(cfg['project']['exp_root']); sg = cfg['experiment']['subgraph']
expected = load_cache_identity(cache_path(cfg, 'screening'))
require_cache_version_compatibility(cfg, expected, 'screening')
rows=[]
for model in MODELS:
    root=exp/model/sg/'screening'; files=sorted(root.glob('*/summary.json')) if root.exists() else []
    valid=[]; rejected=[]
    for f in files:
        try:
            s=json.load(open(f)); require_identity(s, expected, str(f));
            if str(s.get('project_version')) != str(cfg['version']):
                raise ValueError(f'code project version {s.get("project_version")} != {cfg["version"]}')
            valid.append((f,s))
        except Exception as e:
            rejected.append((f,str(e)))
    if not valid:
        msg=f'{model}: no completed screening result matching current Shared Protocol/cache identity.'
        if rejected: msg += f' Rejected {len(rejected)} old/mismatched result(s).'
        raise SystemExit(msg)
    f,s=max(valid,key=lambda x:x[0].stat().st_mtime)
    if int(s['seed']) != int(cfg['experiment']['screening']['seed']):
        raise SystemExit(f'{model}: wrong screening seed in {f}: {s["seed"]}')
    v=s['validation']
    rows.append({'model':model,'run_dir':str(f.parent),'seed':int(s['seed']),'parameters':int(s['parameters']),
      'batch_size':int(s['batch_size']),'best_epoch':int(s['best_epoch']),'recipe_source':s.get('recipe_source'),
      'code_project_version':s.get('project_version'),'shared_contract_version':expected['shared_contract_version'],'data_contract_revision':expected['data_contract_revision'],'time_feature_revision':expected['time_feature_revision'],'time_feature_fingerprint':expected['time_feature_fingerprint'],'protocol_stage_hash':expected['protocol_stage_hash'],
      'cache_fingerprint':expected['cache_fingerprint'],'train_view_hash':expected['split_view_hashes']['train'],
      'val_view_hash':expected['split_view_hashes']['val'],'val_MAE':v['MAE'],'val_RMSE':v['RMSE'],
      'val_masked_MAE':v['directly_masked_MAE'],'val_masked_RMSE':v.get('directly_masked_RMSE'),'val_support_MAE':v['support_dependent_MAE'],'val_support_RMSE':v.get('support_dependent_RMSE'),
      'val_H12_MAE':v['H12_MAE'],'val_H12_RMSE':v.get('H12_RMSE'),'val_H12_support_MAE':v['H12_support_dependent_MAE'],'val_H12_support_RMSE':v.get('H12_support_dependent_RMSE'),
      'val_inference_ms_per_query':v['inference_ms_per_query']})
df=pd.DataFrame(rows)
if len(df)!=len(MODELS): raise SystemExit(f'Need all {len(MODELS)} completed screening models; found {len(df)}')
# Explicit anti-mixing assertion.
for col in ['shared_contract_version','data_contract_revision','time_feature_revision','time_feature_fingerprint','protocol_stage_hash','cache_fingerprint','train_view_hash','val_view_hash']:
    if df[col].nunique()!=1: raise SystemExit(f'FAIL mixed screening identity in {col}: {df[col].unique().tolist()}')
sel=cfg['selection']; df,shortlist,best=screening_shortlist(df,float(sel['overall_tolerance']),float(sel['support_tie_tolerance_rel']),int(sel['shortlist_k']))
out_csv=exp/'backbone_screening_summary.csv'; df.sort_values(['shortlisted','val_MAE'],ascending=[False,True]).to_csv(out_csv,index=False)
obj={'project_version':cfg.get('version'),'stage':'screening','selection_uses_test':False,'run_identity':expected,
 'best_validation_overall_MAE':best,'candidate_models':MODELS,'shortlist':shortlist,
 'rule':'Overall Val MAE within 2% of best; rank eligible by Val support-dependent MAE; support differences <=0.5% use masked MAE, then inference time and parameters; fill Top-2 by Overall MAE if fewer than two eligible.',
 'rows':df.to_dict(orient='records')}
write_json(exp/'screening_shortlist.json',obj)
print(df.to_string(index=False)); print('\nTOP-2 SHORTLIST:',', '.join(shortlist)); print('PROTOCOL_STAGE_HASH='+expected['protocol_stage_hash']); print('CACHE_FINGERPRINT='+expected['cache_fingerprint']); print('wrote',out_csv)
