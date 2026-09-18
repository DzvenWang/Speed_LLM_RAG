#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, json, sys
import numpy as np, torch
from torch.utils.data import DataLoader

ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe')
ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from presure.utils import load_yaml, write_json, sha256_file
from presure.contract import assert_cache_ready, cache_dir
from presure.data import CacheSplit
from presure.pipeline import art_dir
from presure.backbone import load_patchstg_adapter, CachedTorch
from presure.backbone_hidden import discover_hidden_layers, select_sweep_candidates
from presure.query_error_state import (
    build_physical_transition_features, build_repaired_causal_error_features,
    PHYSICAL_TRANSITION_DIM, RECENT_ERROR_REPAIRED_DIM,
)


def save_physical(path, ds, adj, batch=256):
    shape=(len(ds), ds.x.shape[2], PHYSICAL_TRANSITION_DIM)
    mm=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=shape)
    for a in range(0,len(ds),int(batch)):
        z=slice(a,min(len(ds),a+int(batch)))
        mm[z]=np.clip(build_physical_transition_features(ds.x[z],ds.obs[z],adj),-12,12).astype(np.float16)
    del mm


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',default='development',choices=['development']);ap.add_argument('--force',action='store_true');a=ap.parse_args()
    cfg=load_yaml(ROOT/'configs/presure.yaml')
    if str(cfg.get('version'))!='0.9.5.1': raise RuntimeError(f'expected v0.9.5.1, got {cfg.get("version")}')
    ident=assert_cache_ready(cfg,a.stage); art=art_dir(cfg,a.stage); art.mkdir(parents=True,exist_ok=True); cache=cache_dir(cfg,a.stage)
    for n in ['train_base_pred.npy','val_base_pred.npy','graph_adj.npy','backbone_manifest.json','train_node_features.npy','val_node_features.npy']:
        if not (art/n).exists(): raise RuntimeError(f'missing prerequisite {art/n}; prepare backbone/retrieval artifacts first')
    pc=cfg['query_error_state_diagnostics']; lc=pc['layer_sweep']; rc=pc['recent_error_repair']
    manifest_path=art/'query_error_state_diagnostics_manifest.json'
    expected=['train_recent_error_state_repaired.npy','val_recent_error_state_repaired.npy','train_physical_transition.npy','val_physical_transition.npy','patchstg_hidden_layer_catalog.json']
    if manifest_path.exists() and not a.force and all((art/x).exists() for x in expected):
        old=json.loads(manifest_path.read_text()); bm=json.loads((art/'backbone_manifest.json').read_text())
        if old.get('checkpoint_sha256')==bm.get('checkpoint_sha256') and old.get('cache_fingerprint')==ident.get('cache_fingerprint'):
            print('QUERY ERROR-STATE REPRESENTATION DIAGNOSTIC ARTIFACTS already ready and current');return

    tr=CacheSplit(cache,'train');va=CacheSplit(cache,'val');tb=np.load(art/'train_base_pred.npy',mmap_mode='r');adj=np.load(art/'graph_adj.npy')
    # Keep the v0.9.5 physical representation, but regenerate if absent/stale.
    save_physical(art/'train_physical_transition.npy',tr,adj,int(pc['artifact_batch_size']))
    save_physical(art/'val_physical_transition.npy',va,adj,int(pc['artifact_batch_size']))

    # Repaired chronology: Train features use only earlier Train anchors; Validation
    # features use *Train labels only*, never earlier Validation labels.
    rtr,mtr=build_repaired_causal_error_features(
        tr,tb,target_ds=tr,history=int(cfg['experiment']['history']),
        max_lag_queries=int(rc['max_lag_queries']),max_gap_steps=int(rc['max_gap_steps']))
    rva,mva=build_repaired_causal_error_features(
        tr,tb,target_ds=va,history=int(cfg['experiment']['history']),
        max_lag_queries=int(rc['max_lag_queries']),max_gap_steps=int(rc['max_gap_steps']))
    np.save(art/'train_recent_error_state_repaired.npy',np.clip(rtr,-12,12).astype(np.float16))
    np.save(art/'val_recent_error_state_repaired.npy',np.clip(rva,-12,12).astype(np.float16))

    # Build full PatchSTG activation catalogue from *all modules*, including parent blocks.
    dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    adapter,ckpath,ck,src=load_patchstg_adapter(cfg,a.stage,dev)
    ds=CachedTorch(cache,'train');dl=DataLoader(ds,batch_size=int(lc['discovery_batch_size']),shuffle=False,num_workers=0);sample=next(iter(dl))
    catalog=discover_hidden_layers(adapter,sample,int(tr.x.shape[2]),lc)
    sweep=select_sweep_candidates(catalog,lc)
    catj={'all_candidates':[c.to_dict() for c in catalog],'sweep_candidates':[c.to_dict() for c in sweep],
          'selection_contract':'architecture/forward-activation catalogue only; no Train-dev/Validation labels inspected',
          'v095_reference_layer':'input_st_fc'}
    write_json(art/'patchstg_hidden_layer_catalog.json',catj)
    print(f'PATCHSTG HIDDEN CATALOG all={len(catalog)} sweep={len(sweep)}',flush=True)
    for i,c in enumerate(sweep):
        print(f'  LAYER[{i:02d}] name={c.module_name} call={c.call_ordinal} shape={c.raw_shape} width={c.raw_width} exec={c.execution_index} depth={c.module_depth} leaf={c.is_leaf} score={c.score:.4f}',flush=True)

    write_json(manifest_path,{
        'artifact_type':'query_error_state_representation_diagnostics_v0951','version':'0.9.5.1',
        'cache_fingerprint':ident.get('cache_fingerprint'),'checkpoint':str(ckpath),'checkpoint_sha256':sha256_file(ckpath),'checkpoint_source':src,
        'hidden_catalog_path':str(art/'patchstg_hidden_layer_catalog.json'),'hidden_catalog_count':len(catalog),'hidden_sweep_count':len(sweep),
        'recent_error_train':mtr,'recent_error_val':mva,'recent_error_dim':RECENT_ERROR_REPAIRED_DIM,'physical_transition_dim':PHYSICAL_TRANSITION_DIM,
        'leakage_contract':{
            'hidden_layer_discovery':'current incomplete query + frozen PatchSTG forward only; no labels choose layers',
            'layer_selection':'Train-dev diagnostic objective only; Final Validation cannot select a layer',
            'physical':'current query history/mask/graph only',
            'recent_error_train':'earlier Train forecast outcomes only after target times are realised',
            'recent_error_val':'Train forecast outcomes only; no Validation labels are features',
            'current_query_future_as_input':False,
        },
    })
    print('PREPARE QUERY ERROR-STATE REPRESENTATION DIAGNOSTICS PASS')

if __name__=='__main__': main()
