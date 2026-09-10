#!/usr/bin/env python3
from pathlib import Path
import json, sys
GADI_ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone')
ROOT=GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backbone.identity import load_cache_identity, require_identity, require_cache_version_compatibility
from backbone.utils import load_yaml

cfg=load_yaml(ROOT/'configs/backbone_selection.yaml')
exp=Path(cfg['project']['exp_root']); sg=cfg['experiment']['subgraph']
identity=load_cache_identity(exp/'_cache'/sg/'screening')
require_cache_version_compatibility(cfg,identity,'screening')
for model in ['STAEformer','HimNet','BigST','STDN']:
    r=exp/model/sg/'screening'; good=[]
    for f in sorted(r.glob('*/summary.json')) if r.exists() else []:
        try:
            s=json.loads(f.read_text()); require_identity(s,identity,str(f))
            if int(s['seed'])==int(cfg['experiment']['screening']['seed']) and (f.parent/'best.pt').exists(): good.append(f)
        except Exception:
            pass
    if not good: raise SystemExit(f'FAIL Stage-1A missing current screening result/checkpoint: {model}')
    print('PASS Stage-1A',model,good[-1].parent)
print('STAGE-1A READY; recent-model extension may proceed.')
