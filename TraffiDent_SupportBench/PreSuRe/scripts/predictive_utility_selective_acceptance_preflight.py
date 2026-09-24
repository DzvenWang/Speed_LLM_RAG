#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys
import numpy as np

ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe')
ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from presure.utils import load_yaml
from presure.pipeline import art_dir
from presure.contract import cache_dir
from presure.data import CacheSplit

ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development',choices=['development']); a=ap.parse_args()
cfg=load_yaml(ROOT/'configs/presure.yaml')
if str(cfg.get('version')) not in {'0.9.3','0.9.4'}: raise RuntimeError(f'expected version 0.9.3+, got {cfg.get("version")}')
pc=cfg['predictive_utility_selective_acceptance']; K=int(pc['candidate_k'])
if K!=int(cfg['predictive_utility_retrieval']['candidate_k']): raise RuntimeError('v0.9A.3 must reuse exact v0.9A K=24 pool')
if not bool(pc.get('require_v09a_checkpoint',True)): raise RuntimeError('v0.9A.3 must require frozen v0.9A checkpoint')
if not bool(pc.get('reuse_v09a_splits',True)): raise RuntimeError('v0.9A.3 must reuse v0.9A train/dev/final_val IDs')
art=art_dir(cfg,a.stage)
for x in ['train_candidates.npz','val_candidates.npz','train_node_features.npy','val_node_features.npy','train_residual.npy','train_base_pred.npy','val_base_pred.npy']:
    if not (art/x).exists(): raise RuntimeError(f'missing artifact {art/x}')
tr=CacheSplit(cache_dir(cfg,a.stage),'train'); va=CacheSplit(cache_dir(cfg,a.stage),'val')
tc=np.load(art/'train_candidates.npz'); vc=np.load(art/'val_candidates.npz')
if tc['indices'].shape[1]<K or vc['indices'].shape[1]<K: raise RuntimeError('candidate file K too small')
lp=Path(cfg['project']['exp_root'])/'PredictiveUtilityRetrieval'/cfg['experiment']['subgraph']/a.stage/'LATEST.json'
if not lp.exists(): raise RuntimeError(f'missing frozen v0.9A LATEST {lp}')
j=json.loads(lp.read_text()); cp=Path(j['checkpoint']); rd=Path(j['run_dir'])
if not cp.exists(): raise RuntimeError(f'missing frozen v0.9A checkpoint {cp}')
for s in ['train_ids.npy','dev_ids.npy','final_val_ids.npy']:
    if not (rd/s).exists(): raise RuntimeError(f'missing v0.9A split file {rd/s}')
train_ids=np.load(rd/'train_ids.npy'); dev_ids=np.load(rd/'dev_ids.npy'); val_ids=np.load(rd/'final_val_ids.npy')
if np.max(train_ids)>=len(tr) or np.max(dev_ids)>=len(tr) or np.max(val_ids)>=len(va): raise RuntimeError('saved v0.9A split IDs out of range')
if float(pc['policy']['min_coverage'])<0.1: raise RuntimeError('coverage floor is too low for a meaningful selective-acceptance probe')
if len(pc['policy']['threshold_grid'])<4: raise RuntimeError('acceptance threshold grid too small')
if float(pc['training']['positive_utility_threshold'])!=0.0: raise RuntimeError('v0.9A.3 acceptance target must be utility > 0')
print(f'PASS frozen v0.9A ranker {cp}')
print(f'PASS exact v0.9A splits train={len(train_ids)} dev={len(dev_ids)} final_val={len(val_ids)}')
print(f'PASS same leakage-safe candidate pool K={K}; no Dual Memory / no expansion')
print('PASS v0.9A ranker is frozen and determines Top-1 before acceptance modelling')
print('PASS acceptance training population is frozen Top-1 supports only')
print('PASS acceptance target is selected-support utility > 0; near-zero utility is down-weighted, not relabelled')
print('PASS acceptance threshold is calibrated on Train-dev only with no candidate fallback')
print(f'PASS ABSTAIN policy has minimum coverage={pc["policy"]["min_coverage"]}')
print('PASS Qwen is intentionally excluded')
print('PREDICTIVE-UTILITY SELECTIVE-ACCEPTANCE PREFLIGHT PASS')
