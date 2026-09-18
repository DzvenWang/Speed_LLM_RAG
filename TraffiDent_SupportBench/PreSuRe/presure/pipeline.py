from __future__ import annotations
from pathlib import Path
import json, numpy as np, torch
from .models import NodeM1Model, NodeUsefulnessNet, PairwiseSupportRouter


def art_dir(cfg,stage): return Path(cfg['project']['exp_root'])/'_artifacts'/cfg['experiment']['subgraph']/stage
def run_dir(cfg,variant,stage,stamp,seed): return Path(cfg['project']['exp_root'])/variant/cfg['experiment']['subgraph']/stage/f'{stamp}_seed{seed}'

def load_artifacts(cfg,stage):
    a=art_dir(cfg,stage)
    req=['train_base_pred.npy','val_base_pred.npy','train_residual.npy','train_features.npy','val_features.npy','train_candidates.npz','val_candidates.npz','val_oracle_candidates.npz','feature_norm.npz','node_regions.npy','graph_adj.npy','train_node_features.npy','val_node_features.npy','node_feature_manifest.json','region_manifest.json','backbone_manifest.json']
    miss=[x for x in req if not (a/x).exists()]
    if miss: raise RuntimeError(f'Missing PreSuRe v0.7 artifacts {miss}; run prepare + retrieval first')
    return a

def normalize_features(a,split):
    x=np.load(a/f'{split}_features.npy',mmap_mode='r'); z=np.load(a/'feature_norm.npz'); return ((np.asarray(x,np.float32)-z['mean'])/(z['std']+1e-6)).astype(np.float32)

def node_features(a,split): return np.load(a/f'{split}_node_features.npy',mmap_mode='r')

def load_m1(cfg,stage,device):
    p=Path(cfg['project']['exp_root'])/'M1'/cfg['experiment']['subgraph']/stage/'LATEST.json'
    if not p.exists(): raise RuntimeError('M1 checkpoint required first')
    j=json.loads(p.read_text()); ck=torch.load(j['checkpoint'],map_location=device,weights_only=False)
    if ck.get('model_type')!='node_sensing_m1_v04': raise RuntimeError('M1 checkpoint incompatible with the node-sensing encoder; rerun M1')
    m=NodeM1Model(**ck['model_args']).to(device); m.load_state_dict(ck['state_dict']); m.eval(); return m,ck,j['checkpoint']

def load_m3(cfg,stage,device):
    p=Path(cfg['project']['exp_root'])/'M3'/cfg['experiment']['subgraph']/stage/'LATEST.json'
    if not p.exists(): raise RuntimeError('M3 checkpoint required first')
    j=json.loads(p.read_text()); ck=torch.load(j['checkpoint'],map_location=device,weights_only=False)
    if ck.get('model_type')!='node_usefulness_v05': raise RuntimeError('Stale/non-v0.5 M3 checkpoint; rerun M3')
    m=NodeUsefulnessNet(**ck['model_args']).to(device); m.load_state_dict(ck['state_dict']); m.eval(); return m,ck,j['checkpoint']

def load_m4(cfg,stage,device):
    p=Path(cfg['project']['exp_root'])/'M4'/cfg['experiment']['subgraph']/stage/'LATEST.json'
    if not p.exists(): raise RuntimeError('M4 checkpoint required first')
    j=json.loads(p.read_text()); ck=torch.load(j['checkpoint'],map_location=device,weights_only=False)
    if ck.get('model_type')!='pairwise_support_router_v07': raise RuntimeError('Stale/non-v0.7 M4 checkpoint; rerun M4')
    m=PairwiseSupportRouter(**ck['model_args']).to(device); m.load_state_dict(ck['state_dict']); m.eval(); return m,ck,j['checkpoint']
