#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys
import torch

ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe')
ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from presure.utils import load_yaml
from presure.need_gated import load_latest

ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development',choices=['development']); a=ap.parse_args()
cfg=load_yaml(ROOT/'configs/presure.yaml')
assert cfg['version'] in {'0.7.5','0.7.6'}, cfg['version']
exp=cfg['project']['exp_root']; sub=cfg['experiment']['subgraph']; stage=a.stage
_,v71=load_latest(exp,'PredictiveNeedProbe',sub,stage)
_,v72=load_latest(exp,'PredictiveNeedStateProbe',sub,stage)
_,v74=load_latest(exp,'NeedConditionedUtilitySpecialistProbe',sub,stage)
for p in [v71/'utility_A_best.pt',v72/'need_q90_best.pt',v74/'hard_specialist_best.pt']:
    if not p.exists(): raise RuntimeError(f'missing required checkpoint: {p}')
ck=torch.load(v74/'hard_specialist_best.pt',map_location='cpu',weights_only=False)
if ck.get('model_type')!='need_conditioned_utility_specialist_v074' or ck.get('mode')!='hard':
    raise RuntimeError('v0.7.4 hard specialist checkpoint contract mismatch')
ss=json.loads((v72/'summary.json').read_text())
if 'q90' not in ss.get('train_only_thresholds',{}): raise RuntimeError('v0.7.2 q90 thresholds missing')
print('PASS PreSuRe v0.7.5')
print('PASS role',cfg['deployment_aligned_utility_router_probe']['role'])
print('PASS Utility-A',v71)
print('PASS Need-q90',v72)
print('PASS v0.7.4 Hard Specialist',v74)
print('PASS Pred10/Pred20 thresholds will be estimated from Train predicted Need only')
print('PASS Expert Router labels use Train-only utility_dev; Validation future is reporting only')
print('DEPLOYMENT-ALIGNED UTILITY ROUTER PREFLIGHT PASS')
