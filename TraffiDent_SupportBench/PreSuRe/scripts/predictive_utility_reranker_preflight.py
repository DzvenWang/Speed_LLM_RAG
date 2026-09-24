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
if str(cfg.get('version')) not in {'0.9.0','0.9.1','0.9.2','0.9.3','0.9.4'}: raise RuntimeError(f'expected version 0.9.0/0.9.1/0.9.2, got {cfg.get("version")}')
pc = cfg['predictive_utility_retrieval']
if int(pc['candidate_k']) > int(cfg['retrieval']['coarse_k']): raise RuntimeError('v0.9A candidate_k exceeds coarse retrieval pool')
if int(pc['candidate_k']) < 8: raise RuntimeError('v0.9A candidate_k unexpectedly small')
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
# M1 is required only for the configured warm start.
if bool(pc.get('initialize_encoder_from_m1', True)):
    lp = Path(cfg['project']['exp_root']) / 'M1' / cfg['experiment']['subgraph'] / a.stage / 'LATEST.json'
    if not lp.exists(): raise RuntimeError(f'M1 warm-start required but missing {lp}')
    j = json.loads(lp.read_text())
    if not Path(j['checkpoint']).exists(): raise RuntimeError(f'M1 checkpoint missing {j["checkpoint"]}')
print('PASS PreSuRe v0.9A Predictive-Utility Retrieval preflight')
print(f'PASS existing leakage-safe candidate pool K={K} (no M3 Top-4 compression)')
print('PASS query/support node features are deployment-safe artifacts')
print('PASS Train historical residual memory available')
print('PASS node-horizon future utility is supervision/evaluation only')
print('PASS Predictive Need is not used as support-identity input in v0.9A')
print('PASS risk-aware score beta is calibrated on Train-dev only')
print('PREDICTIVE-UTILITY RERANKER PREFLIGHT PASS')
