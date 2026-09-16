#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

GADI_ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline')
ROOT=GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baseline.registry import MAIN_ORDER, SECONDARY_ORDER
from baseline.utils import load_yaml

ap=argparse.ArgumentParser()
ap.add_argument('--include-secondary',action='store_true')
a=ap.parse_args()
cfg=load_yaml(ROOT/'configs/baseline.yaml')
exp=Path(cfg['project']['exp_root']); sg=cfg['experiment']['subgraph']
methods=list(MAIN_ORDER)+(list(SECONDARY_ORDER) if a.include_secondary else [])
for m in methods:
    base=exp/m/sg/'development'
    cands=[]
    if base.exists():
        for d in base.iterdir():
            sp=d/'summary.json'
            if not sp.exists(): continue
            try:
                sm=json.loads(sp.read_text())
                if str(sm.get('code_project_version',sm.get('project_version'))) != str(cfg['version']): continue
                if not bool(sm.get('selection_valid',True)): continue
                score=float(sm.get('val',{}).get('overall_MAE',float('inf')))
                cands.append((score,d,sm))
            except Exception: pass
    if cands:
        score,d,_=sorted(cands,key=lambda x:(x[0],-x[1].stat().st_mtime))[0]
        print(f'{m:16s} PASS overall_MAE={score:.6f} {d.name}')
    else:
        print(f'{m:16s} MISSING')
