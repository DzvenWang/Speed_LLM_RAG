#!/usr/bin/env python3
from pathlib import Path
import os, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if os.environ.get('TRAFFIDENT_TARGET') not in {'speed','flow'}:
    os.environ['TRAFFIDENT_TARGET']='speed'
from backbone.utils import load_yaml
from backbone.path_pointers import resolve_portable_pointer
from backbone.shared_cache import cache_path
cfg=load_yaml(ROOT/'configs/backbone.yaml')
print('Backbone v'+str(cfg['version']), 'target='+str(cfg['experiment']['target']))
for stage,sev in [('formal',None),('evaluation',20),('evaluation',40),('evaluation',60),('evaluation',90)]:
    try:
        p=cache_path(cfg,stage,sev); tag=stage if sev is None else f'evaluation_s{sev}'
        print(tag,'READY' if (p/'READY.json').exists() else 'MISSING',p)
    except Exception as e:
        print(stage if sev is None else f'evaluation_s{sev}','ERROR',repr(e))
models=('STAEformer','HimNet','BigST','STDN','PatchSTG','STLAformer')
base=Path(cfg['project']['exp_root'])
for model in models:
    formal=base/model/cfg['experiment']['subgraph']/'formal'
    ptr=formal/'LATEST_CHECKPOINT.txt'
    if ptr.exists():
        cp=resolve_portable_pointer(ptr.read_text(),cfg['project']['root'],pointer_parent=formal)
        print('BACKBONE_CHECKPOINT',model,'READY' if cp.is_file() else 'BROKEN_POINTER',cp)
    else:
        cands=sorted(formal.glob('*_seed10/best.pt'),reverse=True) if formal.exists() else []
        print('BACKBONE_CHECKPOINT',model,'AVAILABLE_NO_POINTER' if cands else 'MISSING',cands[0] if cands else formal)
