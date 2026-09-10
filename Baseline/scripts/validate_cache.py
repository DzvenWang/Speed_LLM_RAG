#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, json
import pandas as pd
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline'); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT.parent/'Protocol')); sys.dont_write_bytecode=True
from baseline.utils import load_yaml
from baseline.fingerprint import canonical_sha256, manifest_identity
from baseline.cache_identity import data_config_sha256
from shared_protocol.consumer import load_frozen_stage

p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['development','formal','evaluation'],required=True); a=p.parse_args()
cfg=load_yaml(ROOT/'configs/baseline.yaml'); c=Path(cfg['project']['exp_root'])/'_cache'/cfg['experiment']['subgraph']/a.stage
mf=c/'cache_manifest.json'
if not mf.exists(): raise SystemExit(f'MISSING {mf}')
j=json.loads(mf.read_text())
if int(j.get('cache_schema_version',0))!=2:
    raise SystemExit(f'FAIL cache schema {j.get("cache_schema_version",1)}; run scripts/migrate_cache_v040.py --stage {a.stage} or rebuild cache')
if str(j.get('project_version'))!=str(cfg.get('version')):
    raise SystemExit(f'FAIL cache project version {j.get("project_version")} != {cfg.get("version")}')
if str(j.get('data_config_sha256'))!=data_config_sha256(cfg):
    raise SystemExit('FAIL cache data-config identity mismatch')
expected=canonical_sha256({k:v for k,v in j.items() if k!='cache_fingerprint'})
if expected!=j.get('cache_fingerprint'): raise SystemExit('FAIL cache manifest fingerprint mismatch')
pm,pviews=load_frozen_stage(cfg['project']['protocol_artifact_root'],cfg['experiment']['subgraph'],cfg['protocol']['shared_protocol_name'],a.stage)
if j.get('protocol_stage_hash')!=pm['stage_hash']: raise SystemExit('FAIL cache Protocol stage hash mismatch')
expected_splits={'development':{'train','val'},'formal':{'train','val'},'evaluation':{'test'}}[a.stage]
if set(j['splits'])!=expected_splits: raise SystemExit(f'FAIL split isolation expected={expected_splits} actual={set(j["splits"])}')
for split,info in j['splits'].items():
    if str(j.get('split_protocol_view_hashes',{}).get(split))!=str(pm['splits'][split]['view_hash']): raise SystemExit(f'FAIL {split} Protocol view hash mismatch')
    meta=pd.read_csv(c/f'{split}_manifest.csv'); proto=pviews[split]
    if len(meta)!=int(info['rows']) or meta.view_group_id.astype(str).nunique()!=int(info['unique_groups']): raise SystemExit(f'FAIL {split} count/unique mismatch')
    if manifest_identity(meta)!=info['view_manifest_sha256']: raise SystemExit(f'FAIL {split} cache manifest hash mismatch')
    cols=['view_id','view_group_id','family','severity','distribution']
    left=meta[cols].astype(str).reset_index(drop=True); right=proto[cols].astype(str).reset_index(drop=True)
    if not left.equals(right): raise SystemExit(f'FAIL {split} rows differ from frozen Shared Protocol')
    req=['x_obs_norm','x_full_norm','history_observed','artificial_missing','y_raw','y_norm','future_valid','direct_mask','support_mask','tod_hist','dow_hist']
    for name in req:
        pth=c/f'{split}_{name}.npy'
        if not pth.exists(): raise SystemExit(f'FAIL missing {pth}')
    print('PASS',split,'rows',len(meta),'unique_groups',meta.view_group_id.astype(str).nunique(),'families',meta.family.value_counts().to_dict())
print('PASS shared Protocol stage',pm['stage_hash']); print('PASS cache fingerprint',j['cache_fingerprint']); print('CACHE VALIDATION PASS')
