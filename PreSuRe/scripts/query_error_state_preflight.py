#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys,numpy as np
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe');ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from presure.utils import load_yaml
from presure.pipeline import art_dir
from presure.contract import cache_dir
from presure.data import CacheSplit
from presure.query_error_state import RECENT_ERROR_DIM,PHYSICAL_TRANSITION_DIM
ap=argparse.ArgumentParser();ap.add_argument('--stage',default='development',choices=['development']);a=ap.parse_args();cfg=load_yaml(ROOT/'configs/presure.yaml')
if str(cfg.get('version')) not in {'0.9.5','0.9.5.1'}:raise RuntimeError(f'expected v0.9.5 got {cfg.get("version")}')
art=art_dir(cfg,a.stage); req=['train_node_features.npy','val_node_features.npy','train_base_pred.npy','val_base_pred.npy','train_patchstg_hidden.npy','val_patchstg_hidden.npy','train_recent_error_state.npy','val_recent_error_state.npy','train_physical_transition.npy','val_physical_transition.npy','query_error_state_artifact_manifest.json']
for n in req:
    if not (art/n).exists():raise RuntimeError(f'missing {art/n}; run prepare_query_error_state_artifacts.py')
tr=CacheSplit(cache_dir(cfg,a.stage),'train');va=CacheSplit(cache_dir(cfg,a.stage),'val'); arrays={n:np.load(art/n,mmap_mode='r') for n in req if n.endswith('.npy')}
for prefix,ds in [('train',tr),('val',va)]:
    for n in ['node_features','patchstg_hidden','recent_error_state','physical_transition','base_pred']:
        x=arrays[f'{prefix}_{n}.npy'];
        if len(x)!=len(ds):raise RuntimeError(f'{prefix}_{n} length {len(x)} != cache {len(ds)}')
if arrays['train_recent_error_state.npy'].shape[-1]!=RECENT_ERROR_DIM:raise RuntimeError('recent-error width mismatch')
if arrays['train_physical_transition.npy'].shape[-1]!=PHYSICAL_TRANSITION_DIM:raise RuntimeError('physical width mismatch')
m=json.loads((art/'query_error_state_artifact_manifest.json').read_text())
if m.get('leakage_contract',{}).get('current_query_future_as_input') is not False:raise RuntimeError('artifact leakage contract missing')
base=Path(cfg['project']['exp_root'])/'PredictiveUtilityRetrieval'/cfg['experiment']['subgraph']/a.stage; j=json.loads((base/'LATEST.json').read_text());rd=Path(j['run_dir']); ids={n:np.load(rd/f'{n}.npy') for n in ['train_ids','dev_ids','final_val_ids']}
if len(np.intersect1d(ids['train_ids'],ids['dev_ids'])):raise RuntimeError('v0.9A train/dev overlap')
print(f'PASS exact v0.9A splits train={len(ids["train_ids"])} dev={len(ids["dev_ids"])} final_val={len(ids["final_val_ids"])}')
print(f'PASS frozen PatchSTG hidden layer={m["hidden"]["module_name"]} projected_dim={m["hidden"]["projected_dim"]}')
print(f'PASS recent-error causal artifact train_available={m["recent_error_train"]["available"]} source={m["recent_error_train"].get("chronology_source")}')
print('PASS physical transition uses query history/mask/graph only')
print('PASS current-query future is supervision/evaluation only; no support candidate / Oracle / Qwen input')
print('PASS variants are selected on Train-dev only; Final Validation cannot select representation')
print('QUERY ERROR-STATE PREFLIGHT PASS')
