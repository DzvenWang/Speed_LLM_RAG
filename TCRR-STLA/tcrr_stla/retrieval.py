from __future__ import annotations
from pathlib import Path
import json, math
import numpy as np, torch


def _coarse(values: torch.Tensor, mask: torch.Tensor|None, g: int):
    # values: B,H,N. Return B,F and matching validity mask.
    B,H,N=values.shape
    if H%g: raise ValueError(f'history={H} not divisible by period={g}')
    v=values.view(B,H//g,g,N)
    if mask is None:
        c=v.mean(dim=2); m=torch.ones_like(c)
    else:
        mm=mask.view(B,H//g,g,N); den=mm.sum(dim=2); c=(v*mm).sum(dim=2)/den.clamp_min(1.0); m=den/float(g)
    return c.reshape(B,-1),m.reshape(B,-1)

def _pearson_chunk(q,qm,x,min_valid=8):
    # q,qm [B,F], x [C,F] -> [B,C], weighted Pearson using query-valid dimensions only.
    n=qm.sum(1,keepdim=True).clamp_min(1.0); sq=(q*qm).sum(1,keepdim=True); sq2=((q*q)*qm).sum(1,keepdim=True)
    sx=qm@x.t(); sx2=qm@(x*x).t(); sqx=(q*qm)@x.t()
    cov=sqx-sq*sx/n; vq=(sq2-sq*sq/n).clamp_min(1e-8); vx=(sx2-sx*sx/n).clamp_min(1e-8)
    corr=cov/torch.sqrt(vq*vx); corr=torch.nan_to_num(corr,nan=-1.0,posinf=1.0,neginf=-1.0)
    corr=torch.where((qm.sum(1,keepdim=True)>=float(min_valid)),corr,torch.full_like(corr,-1.0))
    return corr

def build_retrieval_cache(cache_dir: Path, split: str, out_path: Path, periods, top_k: int, coverage_gamma: float, query_batch_size: int, memory_chunk_size: int, min_valid_features: int, device: torch.device, limit: int|None=None, memory_cache_dir: Path|None=None, memory_limit: int|None=None):
    memory_cache_dir=Path(memory_cache_dir) if memory_cache_dir is not None else Path(cache_dir); cache_dir=Path(cache_dir)
    memory_x=np.load(memory_cache_dir/'train_x_full_norm.npy',mmap_mode='r'); query_x=np.load(cache_dir/f'{split}_x_obs_norm.npy',mmap_mode='r'); query_m=np.load(cache_dir/f'{split}_history_observed.npy',mmap_mode='r')
    import pandas as pd
    mem_meta=pd.read_csv(memory_cache_dir/'train_manifest.csv'); qry_meta=pd.read_csv(cache_dir/f'{split}_manifest.csv')
    Q=len(qry_meta) if limit is None else min(int(limit),len(qry_meta)); M=len(mem_meta) if memory_limit is None else min(int(memory_limit),len(mem_meta)); mem_meta=mem_meta.iloc[:M].reset_index(drop=True); H=query_x.shape[1]
    indices=np.empty((Q,len(periods),top_k),np.int32); scores=np.empty((Q,len(periods),top_k),np.float32); coverage=np.empty((Q,len(periods)),np.float32)
    mem_starts=torch.as_tensor(mem_meta.window_start.astype(np.int64).to_numpy(),device=device); qry_starts=np.asarray(qry_meta.window_start.astype(np.int64).to_numpy()[:Q]); span=H+int(np.load(memory_cache_dir/'train_y_norm.npy',mmap_mode='r').shape[1])
    for pi,g in enumerate(periods):
        print(f'RETRIEVAL period={g} split={split} Q={Q} M={M}',flush=True)
        # Build full-memory coarse features once on CPU to avoid duplicate transformation per query batch.
        mem_feats=[]
        for ms in range(0,M,memory_chunk_size):
            mx=torch.from_numpy(np.array(memory_x[ms:min(ms+memory_chunk_size,M)],np.float32,copy=True))
            mf,_=_coarse(mx,None,int(g)); mem_feats.append(mf)
        mem_feat=torch.cat(mem_feats,0)
        for qs in range(0,Q,query_batch_size):
            qe=min(qs+query_batch_size,Q); q=torch.from_numpy(np.array(query_x[qs:qe],np.float32,copy=True)).to(device); qm=torch.from_numpy(np.array(query_m[qs:qe],np.float32,copy=True)).to(device)
            qf,qmf=_coarse(q,qm,int(g)); cov=qmf.mean(1); coverage[qs:qe,pi]=cov.detach().cpu().numpy()
            best_s=torch.full((qe-qs,top_k),-float('inf'),device=device); best_i=torch.full((qe-qs,top_k),-1,dtype=torch.long,device=device)
            for ms in range(0,M,memory_chunk_size):
                me=min(ms+memory_chunk_size,M); xf=mem_feat[ms:me].to(device,non_blocking=True); s=_pearson_chunk(qf,qmf,xf,min_valid_features)
                s=s*cov.unsqueeze(1).clamp_min(1e-6).pow(float(coverage_gamma))
                # Exclude temporal overlap for every split; safer than assuming split-boundary non-overlap.
                qst=torch.as_tensor(qry_starts[qs:qe],device=device); mst=mem_starts[ms:me]
                overlap=(mst.unsqueeze(0)<(qst+span).unsqueeze(1)) & (qst.unsqueeze(1)<(mst+span))
                s=s.masked_fill(overlap,-float('inf'))
                cand_s=torch.cat([best_s,s],1); cand_i=torch.cat([best_i,torch.arange(ms,me,device=device).unsqueeze(0).expand(qe-qs,-1)],1)
                best_s,pos=torch.topk(cand_s,k=top_k,dim=1); best_i=torch.gather(cand_i,1,pos)
            if (best_i<0).any(): raise RuntimeError(f'No valid retrieval candidate for split={split} period={g} rows {qs}:{qe}')
            indices[qs:qe,pi]=best_i.cpu().numpy().astype(np.int32); scores[qs:qe,pi]=best_s.cpu().numpy().astype(np.float32)
        del mem_feat
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    out_path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out_path,indices=indices,scores=scores,coverage=coverage,periods=np.asarray(periods,np.int16),split=np.asarray(split),top_k=np.int32(top_k))
    meta={'split':split,'rows':Q,'memory_rows':M,'periods':[int(x) for x in periods],'top_k':int(top_k),'coverage_gamma':float(coverage_gamma),'overlap_span':int(span)}
    out_path.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8'); print(f'RETRIEVAL_CACHE_PASS {out_path}',flush=True)
    return meta
