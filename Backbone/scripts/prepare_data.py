#!/usr/bin/env python3
from pathlib import Path
import argparse, sys
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone')
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT.parent/'Protocol')); sys.path.insert(0,str(ROOT.parent/'TraffiDent_SupportBench')); sys.dont_write_bytecode=True
from backbone.data import load_protocol_views, materialize_split, resolve_time_features
from backbone.utils import load_yaml, ensure_dir, log, write_json
from shared_protocol.utils import canonical_sha256, sha256_file
from supportbench.dataset import SupportBench

p=argparse.ArgumentParser(); p.add_argument('--config',default=str(ROOT/'configs/backbone_selection.yaml')); p.add_argument('--stage',choices=['screening','formal','evaluation'],required=True); p.add_argument('--force',action='store_true'); a=p.parse_args()
cfg=load_yaml(a.config); sg=cfg['experiment']['subgraph']; out=Path(cfg['project']['exp_root'])/'_cache'/sg/a.stage; ensure_dir(out)
marker=out/'READY.json'
if marker.exists() and not a.force:
    log(f'cache already ready: {marker}'); raise SystemExit(0)
if a.stage=='evaluation' and not (Path(cfg['project']['exp_root'])/'FROZEN_BACKBONE.json').exists(): raise RuntimeError('Refusing evaluation cache before FROZEN_BACKBONE.json exists.')
pstage,pmanifest,sets=load_protocol_views(cfg,a.stage)
# Protocol owns sampling; Backbone only asserts expected consumer sizes.
asserts=cfg.get('protocol_assertions',{}).get(a.stage,{})
for split in ('train','val','test'):
    key=f'{split}_rows'
    if key in asserts and split in sets and len(sets[split])!=int(asserts[key]):
        raise RuntimeError(f'Protocol assertion failed for {a.stage}/{split}: expected {asserts[key]} rows, got {len(sets[split])}')
b=SupportBench(sg,root=cfg['project']['benchmark_root']); tod,dow,source=resolve_time_features(b); fill=float(cfg['experiment']['normalized_fill_value'])
for split,df in sets.items():
    log(f'{a.stage}/{split} <= Protocol/{pstage}: {len(df)} views unique_groups={df.view_group_id.astype(str).nunique()} families={df.family.value_counts().to_dict()}')
    materialize_split(b,df,out,split,tod,dow,fill)
speed=b.manifest['primary_channel']; st=b.stats['channels'][speed]
summary={
    'project_version':cfg['version'],'backbone_stage':a.stage,'protocol_stage':pstage,'protocol_stage_hash':pmanifest['stage_hash'],
    'protocol_name':cfg['protocol']['name'],'protocol_version':cfg['protocol']['version'],
    'split_view_hashes':{k:str(pmanifest['splits'][k]['view_hash']) for k in sets},
    'subgraph':sg,'sizes':{k:len(v) for k,v in sets.items()},'unique_groups':{k:int(v.view_group_id.astype(str).nunique()) for k,v in sets.items()},
    'families':{k:{str(x):int(y) for x,y in v.family.value_counts().to_dict().items()} for k,v in sets.items()},
    'speed_mean':float(st['mean']),'speed_std':float(st['std']),'time_feature_source':source,
    'config_sha256':sha256_file(a.config),
}
summary['cache_fingerprint']=canonical_sha256(summary)
write_json(out/'cache_manifest.json',summary); write_json(marker,{'cache_fingerprint':summary['cache_fingerprint'],'protocol_stage_hash':pmanifest['stage_hash']})
log(f'cache ready: {out} protocol_hash={pmanifest["stage_hash"]} cache={summary["cache_fingerprint"]}')
