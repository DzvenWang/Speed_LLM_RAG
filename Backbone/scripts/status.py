#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from backbone.utils import load_yaml, repository_root
from backbone.shared_cache import cache_path, shared_contract_config, common_root
common_root()
from experiment_common.cache import evaluation_access_granted
cfg=load_yaml(ROOT/'configs/backbone.yaml'); print('Backbone v'+cfg['version']); print('dataset',cfg['experiment']['subgraph'])
for stage in ['development','formal']:
    try:
        p=cache_path(cfg,stage); print(stage,'READY' if (p/'READY.json').exists() else 'MISSING',p)
    except Exception as e: print(stage,'ERROR',repr(e))
for sev in cfg['experiment']['evaluation_severities']:
    try:
        p=cache_path(cfg,'evaluation',sev); sc=shared_contract_config(cfg,'evaluation',sev); print(f'evaluation s{sev}','READY' if (p/'READY.json').exists() else 'MISSING','ACCESS='+('GRANTED' if evaluation_access_granted(sc) else 'LOCKED'),p)
    except Exception as e: print(f'evaluation s{sev}','ERROR',repr(e))
f=Path(cfg['project']['exp_root'])/'FROZEN_BACKBONE.json'; print('FROZEN_BACKBONE','YES' if f.exists() else 'NO',f)
