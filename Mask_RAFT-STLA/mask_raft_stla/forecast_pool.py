"""v2.3: deterministic forecast-conditioned shortlisting, using legal Train futures.

Query forecast uses the initial frozen STLA and *observed* query history only.
Candidate future used for indexing is historical, train-split ground truth.
Neither validation nor test targets are read here.
"""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
import numpy as np
import torch
from .config import file_sha256


def select_unique(ids, hist, coverage, forecast_scores, k, history_weight=.55):
    """Rank a multi-resolution pool, deduplicate memory IDs, retain diverse sources.

    Numpy reference implementation makes selection independent of GPU batch grouping.
    """
    ids=np.asarray(ids);hist=np.asarray(hist);coverage=np.asarray(coverage)
    score=history_weight*hist+(1.-history_weight)*forecast_scores
    out_i=[];out_s=[];out_c=[]
    for b in range(ids.shape[0]):
        # Stable sort makes equal scores deterministic across runs.
        order=np.argsort(-score[b],kind='stable')
        seen=set(); rows=[]
        for j in order:
            ident=int(ids[b,j])
            if ident < 0 or ident in seen:continue
            seen.add(ident);rows.append(int(j))
            if len(rows)==k:break
        if len(rows)<k:
            raise RuntimeError(f'Insufficient distinct legal retrieval cases: {len(rows)} < {k}')
        out_i.append(ids[b,rows]);out_s.append(score[b,rows]);out_c.append(coverage[b,rows])
    return np.asarray(out_i,np.int32),np.asarray(out_s,np.float32),np.asarray(out_c,np.float32)


def prepare_forecast_pool(cfg, formal, query_root, split, raw_path, backbone_path, device, *, smoke=False):
    """Cache selected indices for replay; exact frozen teacher and files fingerprinted."""
    from .stla import load_pretrained_stla, MaskAwareSTLAformer
    formal=Path(formal);query_root=Path(query_root);raw_path=Path(raw_path)
    out=raw_path.with_name(raw_path.stem+'.forecast_v23.npz')
    expected={'schema':23,'raw_sha256':file_sha256(raw_path),
              'backbone_sha256':file_sha256(backbone_path),
              'query_x':file_sha256(query_root/f'{split}_x_obs_norm.npy'),
              'query_mask':file_sha256(query_root/f'{split}_history_observed.npy'),
              'train_future':file_sha256(formal/'train_y_norm.npy'),
              'selected':int(cfg['retrieval']['final_candidates']),
              'score_weight':float(cfg['retrieval']['history_similarity_weight']),
              'split':split}
    meta=out.with_suffix('.json')
    if out.is_file() and meta.is_file() and json.loads(meta.read_text())==expected:
        print(f'FORECAST_POOL_HIT split={split} path={out}',flush=True);return out
    base,_=load_pretrained_stla(Path(cfg['project']['root']),Path(backbone_path),device)
    teacher=MaskAwareSTLAformer(base).to(device).eval()
    for p in teacher.parameters():p.requires_grad_(False)
    with np.load(raw_path,allow_pickle=False) as z:
        source_ids=np.asarray(z['indices'],np.int64)
        source_scores=np.asarray(z['scores'],np.float32)
        cov=np.asarray(z['coverage'],np.float32)
    count=24 if smoke else len(source_ids)
    count=min(count,len(source_ids)); pool=source_ids.shape[1]*source_ids.shape[2]
    futures=np.load(formal/'train_y_norm.npy',mmap_mode='r')
    x=np.load(query_root/f'{split}_x_obs_norm.npy',mmap_mode='r')
    mask=np.load(query_root/f'{split}_history_observed.npy',mmap_mode='r')
    tod=np.load(query_root/f'{split}_tod_hist.npy',mmap_mode='r')
    dow=np.load(query_root/f'{split}_dow_hist.npy',mmap_mode='r')
    k=int(cfg['retrieval']['final_candidates']);bw=float(cfg['retrieval']['history_similarity_weight'])
    selected=np.empty((count,k),np.int32);scores=np.empty((count,k),np.float32);covers=np.empty((count,k),np.float32)
    bs=int(cfg['retrieval'].get('forecast_pool_batch_size',16))
    with torch.inference_mode():
        for start in range(0,count,bs):
            stop=min(count,start+bs);qs=torch.as_tensor(np.array(x[start:stop],copy=True),device=device,dtype=torch.float32)
            ms=torch.as_tensor(np.array(mask[start:stop],copy=True),device=device,dtype=torch.float32)
            ts=torch.as_tensor(np.array(tod[start:stop],copy=True),device=device,dtype=torch.long)
            ds=torch.as_tensor(np.array(dow[start:stop],copy=True),device=device,dtype=torch.long)
            pred=teacher(qs,ms,ts,ds)[0].float()
            ids=source_ids[start:stop].reshape(stop-start,pool)
            cy=torch.as_tensor(np.array(futures[ids],copy=True),device=device,dtype=torch.float32)
            # Per-case centered forecast shape: focus on future evolution rather than scale.
            qv=pred.flatten(1);yv=cy.flatten(2)
            qz=qv-qv.mean(-1,keepdim=True);yz=yv-yv.mean(-1,keepdim=True)
            dot=(qz[:,None]*yz).sum(-1)
            sim=dot/(qz.norm(dim=-1)[:,None]*yz.norm(dim=-1)+1e-6)
            sim=torch.nan_to_num(sim,nan=-1.).clamp(-1,1).cpu().numpy()
            ci,cs,cc=select_unique(ids,source_scores[start:stop].reshape(stop-start,pool),
                  np.repeat(cov[start:stop,:,None],source_ids.shape[-1],axis=2).reshape(stop-start,pool),sim,k,bw)
            selected[start:stop]=ci;scores[start:stop]=cs;covers[start:stop]=cc
            if start % max(bs*100,bs)==0:print(f'FORECAST_POOL_PROGRESS split={split} {stop}/{count}',flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_name(out.name+'.tmp.npz')
    np.savez_compressed(tmp,indices=selected,scores=scores,coverage=covers,version=np.int32(23))
    os.replace(tmp,out);meta.write_text(json.dumps(expected,indent=2))
    print(f'FORECAST_POOL_PASS split={split} selected={k} path={out}',flush=True)
    return out
