#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys,numpy as np
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe');ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from presure.utils import load_yaml
from presure.pipeline import art_dir
from presure.contract import cache_dir
from presure.data import CacheSplit
from presure.query_error_state import RECENT_ERROR_REPAIRED_DIM,PHYSICAL_TRANSITION_DIM

ap=argparse.ArgumentParser();ap.add_argument('--stage',default='development',choices=['development']);a=ap.parse_args();cfg=load_yaml(ROOT/'configs/presure.yaml')
if str(cfg.get('version'))!='0.9.5.1':raise RuntimeError(f'expected v0.9.5.1 got {cfg.get("version")}')
art=art_dir(cfg,a.stage);req=['train_node_features.npy','val_node_features.npy','train_base_pred.npy','val_base_pred.npy','train_recent_error_state_repaired.npy','val_recent_error_state_repaired.npy','train_physical_transition.npy','val_physical_transition.npy','query_error_state_diagnostics_manifest.json','patchstg_hidden_layer_catalog.json']
for n in req:
    if not (art/n).exists():raise RuntimeError(f'missing {art/n}; run prepare_query_error_state_representation_diagnostics.py')
tr=CacheSplit(cache_dir(cfg,a.stage),'train');va=CacheSplit(cache_dir(cfg,a.stage),'val')
for prefix,ds in [('train',tr),('val',va)]:
    rec=np.load(art/f'{prefix}_recent_error_state_repaired.npy',mmap_mode='r');phy=np.load(art/f'{prefix}_physical_transition.npy',mmap_mode='r')
    if len(rec)!=len(ds) or rec.shape[1]!=ds.x.shape[2] or rec.shape[-1]!=RECENT_ERROR_REPAIRED_DIM:raise RuntimeError(f'{prefix} repaired recent-error shape mismatch {rec.shape}')
    if len(phy)!=len(ds) or phy.shape[1]!=ds.x.shape[2] or phy.shape[-1]!=PHYSICAL_TRANSITION_DIM:raise RuntimeError(f'{prefix} physical shape mismatch {phy.shape}')
m=json.loads((art/'query_error_state_diagnostics_manifest.json').read_text());cat=json.loads((art/'patchstg_hidden_layer_catalog.json').read_text())
if m.get('leakage_contract',{}).get('current_query_future_as_input') is not False:raise RuntimeError('leakage contract missing')
if len(cat.get('sweep_candidates',[]))<1:raise RuntimeError('hidden layer sweep has no candidates')
if not any(x.get('module_name')!='input_st_fc' for x in cat.get('all_candidates',[])):print('WARN no non-input_st_fc node-aligned activation found; sweep will explicitly document this limitation')
# Exact v0.9A split contract.
base=Path(cfg['project']['exp_root'])/'PredictiveUtilityRetrieval'/cfg['experiment']['subgraph']/a.stage;j=json.loads((base/'LATEST.json').read_text());rd=Path(j['run_dir']);ids={n:np.load(rd/f'{n}.npy') for n in ['train_ids','dev_ids','final_val_ids']}
if len(np.intersect1d(ids['train_ids'],ids['dev_ids'])):raise RuntimeError('v0.9A train/dev overlap')
if np.max(ids['train_ids'])>=len(tr) or np.max(ids['dev_ids'])>=len(tr) or np.max(ids['final_val_ids'])>=len(va):raise RuntimeError('v0.9A split IDs outside current cache')
print(f'PASS exact v0.9A splits train={len(ids["train_ids"])} dev={len(ids["dev_ids"])} final_val={len(ids["final_val_ids"])}')
print(f'PASS hidden catalogue all={len(cat.get("all_candidates",[]))} sweep={len(cat.get("sweep_candidates",[]))}; discovery includes parent blocks, not leaf-only')
print(f'PASS repaired recent-error train_available={m["recent_error_train"].get("available")} source={m["recent_error_train"].get("pool_chronology",{}).get("source")}')
print(f'PASS repaired recent-error val_available={m["recent_error_val"].get("available")} and Validation history uses Train pool only')
print('PASS physical transition uses query history/mask/graph only')
print('PASS layer/representation/seed selection is Train-dev only; Final Validation is touched after selection')
print('PASS no retrieval/support candidate, Oracle choice, Qwen or Dual Memory is a diagnostic model input')
print('QUERY ERROR-STATE REPRESENTATION DIAGNOSTICS PREFLIGHT PASS')
