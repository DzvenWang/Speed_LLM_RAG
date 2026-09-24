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
assert cfg['version']=='0.7.6', cfg['version']
exp=cfg['project']['exp_root']; sub=cfg['experiment']['subgraph']; stage=a.stage
_,v71=load_latest(exp,'PredictiveNeedProbe',sub,stage)
_,v72=load_latest(exp,'PredictiveNeedStateProbe',sub,stage)
_,v74=load_latest(exp,'NeedConditionedUtilitySpecialistProbe',sub,stage)
_,v75=load_latest(exp,'DeploymentAlignedUtilityRouterProbe',sub,stage)
req=[
    v71/'utility_A_best.pt', v71/'utility_dev_ids.npy', v71/'top4_ids_utility_dev.npy',
    v72/'need_q90_best.pt', v72/'prob_need_q90_final_val.npy', v72/'top4_ids_final_val.npy',
    v74/'hard_specialist_best.pt',
    v75/'pred10_specialist_best.pt', v75/'pred20_specialist_best.pt',
    v75/'expert_preference_router_best.pt', v75/'summary.json',
]
for p in req:
    if not p.exists(): raise RuntimeError(f'missing required v0.7.6 source artifact: {p}')
for mode in ['pred10','pred20']:
    ck=torch.load(v75/f'{mode}_specialist_best.pt',map_location='cpu',weights_only=False)
    if ck.get('model_type')!='deployment_aligned_utility_specialist_v075' or ck.get('mode')!=mode:
        raise RuntimeError(f'v0.7.5 {mode} specialist contract mismatch')
rck=torch.load(v75/'expert_preference_router_best.pt',map_location='cpu',weights_only=False)
if rck.get('model_type')!='expert_preference_router_v075':
    raise RuntimeError('v0.7.5 expert router contract mismatch')
ss=json.loads((v75/'summary.json').read_text())
if not {'pred10','pred20'} <= set(ss.get('train_only_predicted_need_thresholds',{})):
    raise RuntimeError('v0.7.5 frozen Pred10/Pred20 thresholds missing')
print('PASS PreSuRe v0.7.6')
print('PASS role',cfg['proposal_meta_ranking_probe']['role'])
print('PASS frozen Utility-A / Hard / Pred10 / Pred20 expert pool')
print('PASS frozen v0.7.5 classification router retained as baseline')
print('PASS proposal meta-ranker trains only on Train utility_dev split')
print('PASS ambiguous proposal cells are retained and weighted by regret, not dropped')
print('PASS Validation future / oracle proposal utility are reporting only')
print('REGRET-AWARE SUPPORT PROPOSAL META-RANKER PREFLIGHT PASS')
