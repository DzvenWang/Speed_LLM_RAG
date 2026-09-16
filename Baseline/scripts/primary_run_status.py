#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path

GADI_ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline')
ROOT=GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baseline.utils import load_yaml
from baseline.run_identity import load_cache_identity, require_same_identity
from baseline.shared_cache import cache_path

cfg=load_yaml(ROOT/'configs/baseline.yaml')
exp=Path(cfg['project']['exp_root'])
sg=cfg['experiment']['subgraph']
expected=load_cache_identity(cache_path(cfg,'formal'),cfg['version'])

print(f"Primary Baseline Formal status v{cfg['version']} dataset={sg}")
for m in cfg['formal_selection_methods']:
    seeds=[int(x) for x in cfg['methods'][m].get('formal_seeds',cfg['experiment']['formal_seeds'])]
    for seed in seeds:
        base=exp/m/sg/'formal'; accepted=[]; rejected=[]
        if base.exists():
            for d in base.glob(f'*_seed{seed}'):
                sp=d/'summary.json'; cp=d/'best.pt'
                if not (sp.exists() and cp.exists()):
                    continue
                try:
                    sm=json.loads(sp.read_text())
                    require_same_identity(sm,expected,label=str(sp))
                    ver=str(sm.get('code_project_version',sm.get('project_version')))
                    if ver != str(cfg['version']): raise ValueError(f'wrong code version {ver}')
                    if str(sm.get('status',''))!='COMPLETE': raise ValueError('not COMPLETE')
                    if not bool(sm.get('selection_valid',False)): raise ValueError('selection_valid=false')
                    score=float(sm.get('val',{}).get('overall_MAE',float('inf')))
                    accepted.append((score,d,sm))
                except Exception as e:
                    rejected.append((d,str(e)))
        if accepted:
            score,d,_=sorted(accepted,key=lambda x:(x[0],-x[1].stat().st_mtime))[0]
            print(f'PASS    {m:14s} seed={seed:2d} overall_MAE={score:.6f} run={d.name}')
        else:
            print(f'MISSING {m:14s} seed={seed:2d} rejected={len(rejected)}')
            for d,why in rejected[-2:]: print(f'  reject {d.name}: {why}')
