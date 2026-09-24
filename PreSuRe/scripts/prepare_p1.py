#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from presure.p1_config import load_config
from presure.contract import assert_cache_ready, cache_dir
from presure.backbone import CachedTorch, load_patchstg_adapter
from presure.patchstg_deep import make_patchstg_inputs, forward_with_deep_states
from presure.p1_memory import (history_features, fit_pca_from_memmap, project_prehead, pooled_key,
    causal_topk, causal_random_k, aggregate_support)
from presure.p1_paths import artifact_dir
from presure.utils import ensure_dir, log, write_json, seed_all


def _starts(cache: Path, split: str) -> np.ndarray:
    df=pd.read_csv(cache/f'{split}_manifest.csv')
    for c in ('window_start','start','window_start_idx','start_idx'):
        if c in df.columns:
            v=pd.to_numeric(df[c],errors='raise').to_numpy(np.int64)
            return v
    raise RuntimeError(f'{split}_manifest.csv has no numeric window start column; columns={list(df.columns)}')


def _loader(cache, split, batch, runtime):
    ds=CachedTorch(cache,split); nw=int(runtime.get('num_workers',0)); pin=bool(runtime.get('pin_memory',True)) and torch.cuda.is_available()
    kw={'num_workers':nw,'pin_memory':pin,'drop_last':False,'shuffle':False}
    if nw>0: kw.update(prefetch_factor=int(runtime.get('prefetch_factor',2)),persistent_workers=bool(runtime.get('persistent_workers',True)))
    return ds,DataLoader(ds,batch_size=int(batch),**kw)


def _infer_split(adapter, cache, split, out, cfg, device):
    ds,dl=_loader(cache,split,cfg['artifacts']['backbone_batch_size'],cfg['runtime'])
    raw_path=out/f'{split}_prehead_raw_tmp.npy'
    base_path=out/f'{split}_base_pred_norm.npy'
    raw_mm=base_mm=None; offset=0; latent_dim=None
    atol=float(cfg['artifacts']['equivalence_atol'])
    with torch.no_grad():
        for bi,b in enumerate(dl,1):
            x,te=make_patchstg_inputs(b,device)
            pred,states=forward_with_deep_states(adapter.model,x,te,capture_layers=())
            pre=states['prehead_node']
            # Verify our exposed-state forward is exactly the same frozen PatchSTG path.
            ref=adapter.eval_forward(b)
            diff=float((pred-ref).abs().max().item())
            if diff>atol: raise RuntimeError(f'PatchSTG exposed-state replay mismatch split={split} batch={bi} max_abs={diff} > {atol}')
            p=pred.detach().cpu().numpy().astype(np.float32); h=pre.detach().cpu().numpy().astype(np.float32)
            if raw_mm is None:
                latent_dim=int(h.shape[-1]); raw_mm=np.lib.format.open_memmap(raw_path,mode='w+',dtype=np.float16,shape=(len(ds),h.shape[1],h.shape[2]))
                base_mm=np.lib.format.open_memmap(base_path,mode='w+',dtype=np.float32,shape=(len(ds),p.shape[1],p.shape[2]))
            n=len(p); raw_mm[offset:offset+n]=h.astype(np.float16); base_mm[offset:offset+n]=p; offset+=n
            if bi%50==0: log(f'P1 PatchSTG cache {split}: {offset}/{len(ds)} rows prehead={tuple(h.shape)}')
    del raw_mm,base_mm
    return raw_path,latent_dim


def _compress_split(out, split, mean, comp, chunk=128):
    raw=np.load(out/f'{split}_prehead_raw_tmp.npy',mmap_mode='r'); Q,N,_=raw.shape; d=comp.shape[0]
    zmm=np.lib.format.open_memmap(out/f'{split}_query_latent.npy',mode='w+',dtype=np.float16,shape=(Q,N,d))
    kmm=np.lib.format.open_memmap(out/f'{split}_retrieval_key.npy',mode='w+',dtype=np.float32,shape=(Q,d))
    for a in range(0,Q,int(chunk)):
        b=min(a+int(chunk),Q); z=project_prehead(np.asarray(raw[a:b],np.float32),mean,comp)
        zmm[a:b]=z.astype(np.float16); kmm[a:b]=pooled_key(z)
    del zmm,kmm,raw


def _history_artifact(cache,split,out,chunk=256):
    x=np.load(cache/f'{split}_x_obs_norm.npy',mmap_mode='r'); m=np.load(cache/f'{split}_history_observed.npy',mmap_mode='r')
    Q,H,N=x.shape; mm=np.lib.format.open_memmap(out/f'{split}_query_history_feat.npy',mode='w+',dtype=np.float16,shape=(Q,N,6))
    for a in range(0,Q,int(chunk)):
        b=min(a+int(chunk),Q); mm[a:b]=history_features(np.asarray(x[a:b],np.float32),np.asarray(m[a:b],bool)).astype(np.float16)
    del mm


def _future_memory(cache,out,mean,std):
    y=np.load(cache/'train_y_raw.npy',mmap_mode='r'); valid=np.load(cache/'train_future_valid.npy',mmap_mode='r')
    hist=np.load(out/'train_query_history_feat.npy',mmap_mode='r'); Q,P,N=y.shape
    evo=np.lib.format.open_memmap(out/'train_memory_future_evolution.npy',mode='w+',dtype=np.float16,shape=(Q,P,N))
    mask=np.lib.format.open_memmap(out/'train_memory_future_mask.npy',mode='w+',dtype=np.uint8,shape=(Q,P,N))
    chunk=256
    for a in range(0,Q,chunk):
        b=min(a+chunk,Q); yn=(np.asarray(y[a:b],np.float32)-float(mean))/float(std)
        last=np.asarray(hist[a:b,:,2],np.float32)[:,None,:]
        has=(np.asarray(hist[a:b,:,5],np.float32)>0)[:,None,:]
        vm=np.asarray(valid[a:b],bool)&has
        ee=np.where(vm,yn-last,0.0)
        evo[a:b]=ee.astype(np.float16); mask[a:b]=vm.astype(np.uint8)
    del evo,mask


def _build_split_evidence(cfg,cache,out,split,seed):
    H=int(cfg['experiment']['history']); P=int(cfg['experiment']['horizon']); K=int(cfg['retrieval']['top_k'])
    qk=np.load(out/f'{split}_retrieval_key.npy',mmap_mode='r'); mk=np.load(out/'train_retrieval_key.npy',mmap_mode='r')
    qs=_starts(cache,split); ms=_starts(cache,'train')
    inds,sims,w=causal_topk(qk,mk,qs,ms,H,P,K,float(cfg['retrieval']['temperature']),int(cfg['retrieval']['query_chunk']))
    rinds,rw=causal_random_k(qs,ms,H,P,K,int(seed)+(0 if split=='train' else 10000))
    np.save(out/f'{split}_topk_indices.npy',inds); np.save(out/f'{split}_topk_similarity.npy',sims.astype(np.float16)); np.save(out/f'{split}_topk_weights.npy',w.astype(np.float16)); np.save(out/f'{split}_random_indices.npy',rinds)
    mem_h=np.load(out/'train_query_history_feat.npy',mmap_mode='r'); mem_f=np.load(out/'train_memory_future_evolution.npy',mmap_mode='r'); mem_m=np.load(out/'train_memory_future_mask.npy',mmap_mode='r')
    for pref,ii,ww in [('real',inds,w),('random',rinds,rw)]:
        sh,sf,sc=aggregate_support(ii,ww,mem_h,mem_f,mem_m)
        np.save(out/f'{split}_{pref}_support_history.npy',sh.astype(np.float16)); np.save(out/f'{split}_{pref}_support_future.npy',sf.astype(np.float16)); np.save(out/f'{split}_{pref}_support_future_cov.npy',sc.astype(np.float16))
    nvalid=(inds>=0).sum(axis=1)
    log(f'P1 retrieval {split}: rows={len(qs)} mean_valid_K={nvalid.mean():.2f}/{K} no_history={(nvalid==0).mean():.4f} mean_top1={np.nanmean(np.where(np.isfinite(sims[:,0]),sims[:,0],np.nan)):.4f}')
    return {'rows':len(qs),'mean_valid_k':float(nvalid.mean()),'no_history_fraction':float((nvalid==0).mean())}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development',choices=['development','formal']); ap.add_argument('--force',action='store_true'); args=ap.parse_args()
    cfg=load_config(ROOT); seed=int(cfg['experiment']['seed']); seed_all(seed); cache=cache_dir(cfg,args.stage); ident=assert_cache_ready(cfg,args.stage)
    out=ensure_dir(artifact_dir(cfg,args.stage)); ready=out/'READY.json'
    if ready.exists() and not args.force:
        log(f'P1 artifacts already ready: {ready}'); return
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); log(f'P1 PREPARE stage={args.stage} device={device} cache={cache}')
    adapter,ckpath,ck,source=load_patchstg_adapter(cfg,args.stage,device)
    train_raw,D=_infer_split(adapter,cache,'train',out,cfg,device); val_raw,D2=_infer_split(adapter,cache,'val',out,cfg,device)
    if D!=D2: raise RuntimeError(f'prehead dims differ train={D} val={D2}')
    mean_pca,comp,expl=fit_pca_from_memmap(train_raw,int(cfg['artifacts']['latent_pca_dim']),int(cfg['artifacts']['pca_fit_vectors']),seed)
    np.savez(out/'latent_pca.npz',mean=mean_pca,components=comp,explained_ratio=expl)
    _compress_split(out,'train',mean_pca,comp,int(cfg['artifacts']['projection_chunk'])); _compress_split(out,'val',mean_pca,comp,int(cfg['artifacts']['projection_chunk']))
    _history_artifact(cache,'train',out); _history_artifact(cache,'val',out)
    speed_mean=float(ck['mean']); speed_std=float(ck['std']); _future_memory(cache,out,speed_mean,speed_std)
    tr_info=_build_split_evidence(cfg,cache,out,'train',seed); va_info=_build_split_evidence(cfg,cache,out,'val',seed)
    if bool(cfg['artifacts'].get('cleanup_raw_prehead',True)):
        for p in [out/'train_prehead_raw_tmp.npy',out/'val_prehead_raw_tmp.npy']:
            try: os.remove(p)
            except FileNotFoundError: pass
    meta={'version':cfg['version'],'stage':args.stage,'checkpoint':str(ckpath),'checkpoint_source':source,'speed_mean':speed_mean,'speed_std':speed_std,'prehead_dim':int(D),'latent_pca_dim':int(comp.shape[0]),'pca_explained_sum':float(expl.sum()),'retrieval_top_k':int(cfg['retrieval']['top_k']),'strict_causal_episode_end_before_query':True,'historical_future_mask':'train_future_valid','train_retrieval':tr_info,'val_retrieval':va_info,'cache_identity':ident}
    write_json(ready,meta); log(f'P1 PREPARE PASS READY={ready} pca_explained={expl.sum():.4f}')

if __name__=='__main__': main()
