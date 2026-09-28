#!/usr/bin/env python3
from pathlib import Path
import os, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if os.environ.get('TRAFFIDENT_TARGET') not in {'speed','flow'}:
    os.environ['TRAFFIDENT_TARGET']='speed'
from backbone.utils import load_yaml
from backbone.shared_cache import cache_path
cfg=load_yaml(ROOT/'configs/backbone.yaml')
print('Backbone v'+str(cfg['version']), 'target='+str(cfg['experiment']['target']))
for stage,sev in [('formal',None),('evaluation',20),('evaluation',40),('evaluation',60),('evaluation',90)]:
    try:
        p=cache_path(cfg,stage,sev)
        tag=stage if sev is None else f'evaluation_s{sev}'
        print(tag,'READY' if (p/'READY.json').exists() else 'MISSING',p)
    except Exception as e:
        print(stage if sev is None else f'evaluation_s{sev}','ERROR',repr(e))
f=Path(cfg['project']['exp_root'])/'FROZEN_BACKBONE.json'
print('FROZEN_BACKBONE','YES' if f.exists() else 'NO',f)
