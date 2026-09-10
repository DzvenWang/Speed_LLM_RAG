#!/usr/bin/env python3
"""Migrate v0.3.1 Baseline cache manifest to v0.4.0 schema without rewriting tensors.

The tensor materialization contract is unchanged. This script verifies the old cache against
its own fingerprint, Shared Protocol views, metadata hashes, and required arrays before
rebinding only the cache identity schema. It never changes any *.npy or *_manifest.csv file.
"""
from pathlib import Path
import argparse, json, shutil, sys, time
import pandas as pd
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline'); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT.parent/'Protocol')); sys.path.insert(0,str(ROOT.parent/'TraffiDent_SupportBench')); sys.dont_write_bytecode=True
from baseline.utils import load_yaml, write_json
from baseline.fingerprint import canonical_sha256, manifest_identity
from baseline.cache_identity import data_config_sha256
from shared_protocol.consumer import load_frozen_stage

ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['development','formal'],default='development'); args=ap.parse_args()
cfg=load_yaml(ROOT/'configs/baseline.yaml'); cache=Path(cfg['project']['exp_root'])/'_cache'/cfg['experiment']['subgraph']/args.stage
p=cache/'cache_manifest.json'
if not p.exists(): raise SystemExit(f'MISSING {p}')
old=json.loads(p.read_text())
if int(old.get('cache_schema_version',1))==2 and old.get('project_version')==cfg['version']:
    print('CACHE ALREADY v0.4.0 schema2; nothing to migrate.'); raise SystemExit(0)
allowed=set(map(str,cfg['protocol'].get('development_cache_migration_from',[])))
if str(old.get('project_version')) not in allowed:
    raise SystemExit(f'REFUSE migration from project_version={old.get("project_version")}; allowed={sorted(allowed)}')
old_fp=canonical_sha256({k:v for k,v in old.items() if k!='cache_fingerprint'})
if old_fp!=old.get('cache_fingerprint'): raise SystemExit('REFUSE: old cache fingerprint is invalid')
pm,pviews=load_frozen_stage(cfg['project']['protocol_artifact_root'],cfg['experiment']['subgraph'],cfg['protocol']['shared_protocol_name'],args.stage)
if old.get('protocol_stage_hash')!=pm['stage_hash']: raise SystemExit('REFUSE: Protocol stage hash mismatch')
expected_splits={'development':{'train','val'},'formal':{'train','val'}}[args.stage]
if set(old['splits'])!=expected_splits: raise SystemExit('REFUSE: split set mismatch')
for split,info in old['splits'].items():
    meta=pd.read_csv(cache/f'{split}_manifest.csv'); proto=pviews[split]
    if manifest_identity(meta)!=info['view_manifest_sha256']: raise SystemExit(f'REFUSE: {split} manifest identity mismatch')
    cols=['view_id','view_group_id','family','severity','distribution']
    if not meta[cols].astype(str).reset_index(drop=True).equals(proto[cols].astype(str).reset_index(drop=True)):
        raise SystemExit(f'REFUSE: {split} rows differ from frozen Protocol')
    for name in ['x_obs_norm','x_full_norm','history_observed','artificial_missing','y_raw','y_norm','future_valid','direct_mask','support_mask','tod_hist','dow_hist']:
        if not (cache/f'{split}_{name}.npy').exists(): raise SystemExit(f'REFUSE: missing {split}_{name}.npy')
new={
    'cache_schema_version':2,
    'project_version':cfg['version'],
    'materializer_version':1,
    'stage':args.stage,
    'subgraph':cfg['experiment']['subgraph'],
    'splits':old['splits'],
    'protocol_stage_hash':old['protocol_stage_hash'],
    'split_protocol_view_hashes':old['split_protocol_view_hashes'],
    'data_config_sha256':data_config_sha256(cfg),
    'supportbench_manifest_sha256':old.get('supportbench_manifest_sha256'),
    'migrated_from':str(old.get('project_version')),
}
new['cache_fingerprint']=canonical_sha256({k:v for k,v in new.items() if k!='cache_fingerprint'})
stamp=time.strftime('%Y%m%d_%H%M%S')
backup=cache/f'cache_manifest.v031_before_v040_{stamp}.json'; shutil.copy2(p,backup); write_json(p,new)
print('PASS old fingerprint',old['cache_fingerprint'])
print('PASS Protocol stage',pm['stage_hash'])
print('BACKUP',backup)
print('NEW CACHE_FINGERPRINT='+new['cache_fingerprint'])
print('CACHE MIGRATION PASS; tensor files were not modified.')
