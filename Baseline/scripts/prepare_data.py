#!/usr/bin/env python3
from pathlib import Path
import argparse, sys
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline')
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT.parent/'Protocol')); sys.path.insert(0,str(ROOT.parent/'TraffiDent_SupportBench')); sys.dont_write_bytecode=True
from baseline.utils import load_yaml, ensure_dir, write_json
from baseline.data import load_protocol_views, materialize
from baseline.fingerprint import canonical_sha256, manifest_identity
from baseline.cache_identity import data_config_sha256

p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['development','formal','evaluation'],default='development'); a=p.parse_args()
cfg=load_yaml(ROOT/'configs/baseline.yaml')
if a.stage=='evaluation' and not (Path(cfg['project']['exp_root'])/'FROZEN_BASELINE_PROTOCOL.json').exists():
    raise SystemExit('REFUSE: baseline protocol is not frozen.')
from supportbench.dataset import SupportBench
b=SupportBench(cfg['experiment']['subgraph'],root=cfg['project']['benchmark_root'])
out=Path(cfg['project']['exp_root'])/'_cache'/cfg['experiment']['subgraph']/a.stage; ensure_dir(out)
pm,ms=load_protocol_views(cfg,a.stage)
asserts=cfg.get('protocol_assertions',{}).get(a.stage,{})
for split in ('train','val','test'):
    key=f'{split}_rows'
    if key in asserts and split in ms and len(ms[split])!=int(asserts[key]):
        raise SystemExit(f'Protocol assertion failed {a.stage}/{split}: expected {asserts[key]}, got {len(ms[split])}')
summary={
    'cache_schema_version':2,
    'project_version':cfg['version'],
    'materializer_version':1,
    'stage':a.stage,
    'subgraph':cfg['experiment']['subgraph'],
    'splits':{},
    'protocol_stage_hash':pm['stage_hash'],
    'split_protocol_view_hashes':{k:str(pm['splits'][k]['view_hash']) for k in ms},
    'data_config_sha256':data_config_sha256(cfg),
    'supportbench_manifest_sha256':canonical_sha256(b.manifest),
}
for split,df in ms.items():
    counts={str(k):int(v) for k,v in df.family.value_counts().to_dict().items()}
    print(f'{a.stage}/{split} <= SharedProtocol: {len(df)} views unique_groups={df.view_group_id.astype(str).nunique()} families={counts}',flush=True)
    info=materialize(b,df,out,split)
    summary['splits'][split]={
        'rows':len(df),'unique_groups':int(df.view_group_id.astype(str).nunique()),
        'families':counts,'view_manifest_sha256':manifest_identity(df),**info
    }
summary['cache_fingerprint']=canonical_sha256({k:v for k,v in summary.items() if k!='cache_fingerprint'})
write_json(out/'cache_manifest.json',summary)
print('PROTOCOL_STAGE_HASH='+pm['stage_hash'])
print('CACHE_FINGERPRINT='+summary['cache_fingerprint'])
print('cache ready:',out)
