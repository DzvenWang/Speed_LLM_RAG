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
from presure.features import NODE_SENSE_DIM

ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development',choices=['development']); a=ap.parse_args()
cfg=load_yaml(ROOT/'configs/presure.yaml')
if str(cfg.get('version')) not in {'0.9.4','0.9.5'}: raise RuntimeError(f'expected version 0.9.4+, got {cfg.get("version")}')
pc=cfg['query_future_error_direction_probe']
if not bool(pc.get('reuse_v09a_splits',True)): raise RuntimeError('direction probe must reuse exact v0.9A split IDs')
art=art_dir(cfg,a.stage)
for x in ['train_node_features.npy','val_node_features.npy','train_base_pred.npy','val_base_pred.npy','node_feature_manifest.json','backbone_manifest.json']:
    if not (art/x).exists(): raise RuntimeError(f'missing artifact {art/x}')
tr=CacheSplit(cache_dir(cfg,a.stage),'train'); va=CacheSplit(cache_dir(cfg,a.stage),'val')
nft=np.load(art/'train_node_features.npy',mmap_mode='r'); nfv=np.load(art/'val_node_features.npy',mmap_mode='r')
if nft.shape[0]!=len(tr) or nfv.shape[0]!=len(va): raise RuntimeError('node-feature/cache length mismatch')
if nft.shape[-1] < NODE_SENSE_DIM: raise RuntimeError('node feature width too small')
base=Path(cfg['project']['exp_root'])/'PredictiveUtilityRetrieval'/cfg['experiment']['subgraph']/a.stage
lp=base/'LATEST.json'
if not lp.exists(): raise RuntimeError(f'missing v0.9A LATEST {lp}')
j=json.loads(lp.read_text()); rd=Path(j['run_dir'])
for name in ['train_ids','dev_ids','final_val_ids']:
    p=rd/f'{name}.npy'
    if not p.exists(): raise RuntimeError(f'missing v0.9A split {p}')
train_ids=np.load(rd/'train_ids.npy'); dev_ids=np.load(rd/'dev_ids.npy'); val_ids=np.load(rd/'final_val_ids.npy')
if len(np.intersect1d(train_ids,dev_ids))>0: raise RuntimeError('Train/dev split overlap')
if np.max(train_ids)>=len(tr) or np.max(dev_ids)>=len(tr) or np.max(val_ids)>=len(va): raise RuntimeError('saved split IDs out of range')
qs=pc['thresholds']['evaluation_quantiles']
if [float(x) for x in qs] != [0.5,0.75,0.9]: raise RuntimeError('expected q50/q75/q90 diagnostic thresholds')
if float(pc['thresholds']['train_direction_quantile'])!=0.5: raise RuntimeError('training direction threshold must be Train q50')
print(f'PASS exact v0.9A splits train={len(train_ids)} dev={len(dev_ids)} final_val={len(val_ids)}')
print(f'PASS query node features train={nft.shape} val={nfv.shape}')
print('PASS query-only inputs: incomplete sensing context + frozen PatchSTG forecast; no support/candidate input')
print('PASS direction target Y_future - Y_base is used for Train supervision / diagnostic evaluation only')
print('PASS q50/q75/q90 error thresholds are derived from Train split only, horizon by horizon')
print('PASS Validation future is evaluation-only and does not enter model inputs or threshold calibration')
print('QUERY FUTURE-ERROR DIRECTION PREFLIGHT PASS')
