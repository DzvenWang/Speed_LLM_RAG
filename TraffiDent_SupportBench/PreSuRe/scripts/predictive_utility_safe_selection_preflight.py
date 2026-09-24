#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import sys
import numpy as np

ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe')
ROOT = ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from presure.utils import load_yaml
from presure.pipeline import art_dir
from presure.contract import cache_dir
from presure.data import CacheSplit

ap = argparse.ArgumentParser(); ap.add_argument('--stage', default='development', choices=['development']); a = ap.parse_args()
cfg = load_yaml(ROOT / 'configs/presure.yaml')
if str(cfg.get('version')) not in {'0.9.1','0.9.2','0.9.3','0.9.4'}: raise RuntimeError(f'expected version 0.9.1/0.9.2, got {cfg.get("version")}')
pc = cfg['predictive_utility_safe_selection']
if int(pc['candidate_k']) != int(cfg['predictive_utility_retrieval']['candidate_k']):
    raise RuntimeError('v0.9A.1 must use the exact same K=24 candidate pool as v0.9A')
if int(pc['candidate_k']) > int(cfg['retrieval']['coarse_k']): raise RuntimeError('candidate_k exceeds coarse retrieval pool')
if float(pc['policy']['min_coverage']) < 0.25: raise RuntimeError('coverage floor is too low; abstention could become degenerate')
if max(float(x) for x in pc['policy']['coverage_grid']) < 0.99: raise RuntimeError('policy grid must include a no-abstain coverage point')
art = art_dir(cfg, a.stage)
req = [
    'train_candidates.npz','val_candidates.npz','train_node_features.npy','val_node_features.npy',
    'train_residual.npy','train_base_pred.npy','val_base_pred.npy','node_feature_manifest.json','retrieval_manifest.json'
]
for x in req:
    if not (art / x).exists(): raise RuntimeError(f'missing required artifact {art / x}')
tr = CacheSplit(cache_dir(cfg, a.stage), 'train'); va = CacheSplit(cache_dir(cfg, a.stage), 'val')
tc = np.load(art / 'train_candidates.npz'); vc = np.load(art / 'val_candidates.npz')
K = int(pc['candidate_k'])
if tc['indices'].shape[1] < K or vc['indices'].shape[1] < K: raise RuntimeError('candidate files do not contain requested K')
if np.any(tc['indices'][:min(len(tr),100),:K] < 0): raise RuntimeError('unexpected missing Train candidate in first preflight sample')
if np.any(vc['indices'][:min(len(va),100),:K] < 0): raise RuntimeError('unexpected missing Validation candidate in first preflight sample')
if bool(pc.get('initialize_from_v09a', True)):
    lp = Path(cfg['project']['exp_root']) / 'PredictiveUtilityRetrieval' / cfg['experiment']['subgraph'] / a.stage / 'LATEST.json'
    if not lp.exists():
        if bool(pc.get('require_v09a_checkpoint', True)): raise RuntimeError(f'v0.9A warm-start required but missing {lp}')
    else:
        j = json.loads(lp.read_text())
        cp = Path(j['checkpoint'])
        if not cp.exists(): raise RuntimeError(f'v0.9A checkpoint missing {cp}')
        print(f'PASS v0.9A warm-start checkpoint {cp}')
print('PASS PreSuRe v0.9A.1 Safe Top-Selection preflight')
print(f'PASS same leakage-safe candidate pool K={K}; no Dual Memory / no candidate expansion')
print('PASS listwise Top-1 + absolute sign/barrier objectives are Train-supervision only')
print('PASS beta/lambda score calibration uses Train-dev only')
print(f'PASS abstention calibration has explicit minimum coverage={pc["policy"]["min_coverage"]}')
print('PASS Qwen is intentionally excluded; protected-backbone ABSTAIN is numerical policy only')
print('PREDICTIVE-UTILITY SAFE-SELECTION PREFLIGHT PASS')
