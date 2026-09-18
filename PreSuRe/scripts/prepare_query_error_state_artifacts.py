#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, json, sys
import numpy as np, torch
from torch.utils.data import DataLoader

ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe'); ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from presure.utils import load_yaml, write_json, sha256_file
from presure.contract import assert_cache_ready, cache_dir
from presure.data import CacheSplit
from presure.pipeline import art_dir
from presure.backbone import load_patchstg_adapter, CachedTorch
from presure.backbone_hidden import discover_hidden_layer, infer_hidden_split
from presure.query_error_state import build_physical_transition_features, build_recent_causal_error_features, RECENT_ERROR_DIM, PHYSICAL_TRANSITION_DIM


def save_physical(path, ds, adj, batch=256):
    shape=(len(ds), ds.x.shape[2], PHYSICAL_TRANSITION_DIM)
    mm=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=shape)
    for a in range(0,len(ds),int(batch)):
        z=slice(a,min(len(ds),a+int(batch)))
        mm[z]=np.clip(build_physical_transition_features(ds.x[z],ds.obs[z],adj),-12,12).astype(np.float16)
    del mm


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development',choices=['development']); ap.add_argument('--force',action='store_true'); a=ap.parse_args()
    cfg=load_yaml(ROOT/'configs/presure.yaml')
    if str(cfg.get('version')) not in {'0.9.5','0.9.5.1'}: raise RuntimeError(f'expected v0.9.5, got {cfg.get("version")}')
    ident=assert_cache_ready(cfg,a.stage); art=art_dir(cfg,a.stage); art.mkdir(parents=True,exist_ok=True); cache=cache_dir(cfg,a.stage)
    for n in ['train_base_pred.npy','val_base_pred.npy','graph_adj.npy','backbone_manifest.json','train_node_features.npy','val_node_features.npy']:
        if not (art/n).exists(): raise RuntimeError(f'missing prerequisite {art/n}; run backbone/retrieval artifact preparation first')
    pc=cfg['query_error_state_probe']; hc=pc['hidden_artifact']; cc=pc['recent_error']
    manifest_path=art/'query_error_state_artifact_manifest.json'
    expected=['train_patchstg_hidden.npy','val_patchstg_hidden.npy','train_recent_error_state.npy','val_recent_error_state.npy','train_physical_transition.npy','val_physical_transition.npy']
    if manifest_path.exists() and not a.force and all((art/x).exists() for x in expected):
        old=json.loads(manifest_path.read_text())
        bm=json.loads((art/'backbone_manifest.json').read_text())
        if old.get('checkpoint_sha256')==bm.get('checkpoint_sha256') and old.get('cache_fingerprint')==ident.get('cache_fingerprint'):
            print('QUERY ERROR-STATE ARTIFACTS already ready and current'); return

    tr=CacheSplit(cache,'train'); va=CacheSplit(cache,'val'); tb=np.load(art/'train_base_pred.npy',mmap_mode='r'); vb=np.load(art/'val_base_pred.npy',mmap_mode='r'); adj=np.load(art/'graph_adj.npy')
    # Physical transition: query history only.
    save_physical(art/'train_physical_transition.npy',tr,adj,int(pc['artifact_batch_size']))
    save_physical(art/'val_physical_transition.npy',va,adj,int(pc['artifact_batch_size']))

    # Strictly causal recent backbone error. Disable rather than guess when chronology cannot be proven.
    rtr,mtr=build_recent_causal_error_features(tr,tb,expected_stride=int(cc['expected_stride']),min_stride_confidence=float(cc['min_stride_confidence']),max_lag_queries=int(cc['max_lag_queries']),max_gap_steps=int(cc['max_gap_steps']),step_minutes=int(cc['step_minutes']))
    rva,mva=build_recent_causal_error_features(va,vb,expected_stride=int(cc['expected_stride']),min_stride_confidence=float(cc['min_stride_confidence']),max_lag_queries=int(cc['max_lag_queries']),max_gap_steps=int(cc['max_gap_steps']),step_minutes=int(cc['step_minutes']))
    np.save(art/'train_recent_error_state.npy',np.clip(rtr,-12,12).astype(np.float16)); np.save(art/'val_recent_error_state.npy',np.clip(rva,-12,12).astype(np.float16))

    # Frozen PatchSTG predictive hidden state, selected once and audited in manifest.
    dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); adapter,ckpath,ck,src=load_patchstg_adapter(cfg,a.stage,dev)
    dset=CachedTorch(cache,'train'); dl=DataLoader(dset,batch_size=int(hc['discovery_batch_size']),shuffle=False,num_workers=0); sample=next(iter(dl))
    choice=discover_hidden_layer(adapter,sample,int(tr.x.shape[2]),hc)
    print(f'HIDDEN LAYER selected={choice.module_name} call={choice.call_ordinal} shape={choice.raw_shape} raw_width={choice.raw_width} score={choice.score:.4f}',flush=True)
    outw=min(int(hc['projected_dim']),int(choice.raw_width))
    htr=infer_hidden_split(adapter,cache,'train',choice.module_name,choice.call_ordinal,choice.raw_width,outw,int(hc['projection_seed']),int(hc['batch_size']),int(tr.x.shape[2])); np.save(art/'train_patchstg_hidden.npy',htr)
    hva=infer_hidden_split(adapter,cache,'val',choice.module_name,choice.call_ordinal,choice.raw_width,outw,int(hc['projection_seed']),int(hc['batch_size']),int(tr.x.shape[2])); np.save(art/'val_patchstg_hidden.npy',hva)

    write_json(manifest_path,{
        'artifact_type':'query_error_state_v095','version':'0.9.5','cache_fingerprint':ident.get('cache_fingerprint'),'checkpoint':str(ckpath),'checkpoint_sha256':sha256_file(ckpath),'checkpoint_source':src,
        'hidden':{'module_name':choice.module_name,'call_ordinal':choice.call_ordinal,'raw_shape':choice.raw_shape,'raw_width':choice.raw_width,'projected_dim':outw,'selection_score':choice.score,'selection_rule':'deep node-aligned frozen PatchSTG leaf activation; deterministic projection'},
        'recent_error_train':mtr,'recent_error_val':mva,'recent_error_dim':RECENT_ERROR_DIM,'physical_transition_dim':PHYSICAL_TRANSITION_DIM,
        'leakage_contract':{'hidden':'frozen PatchSTG forward on current incomplete query only','physical':'current query history/mask/graph only','recent_error':'previous-query forecasts only after their target timestamps are realized; disabled if chronology cannot be proven','current_query_future_as_input':False},
    })
    print('PREPARE QUERY ERROR-STATE ARTIFACTS PASS')

if __name__=='__main__': main()
