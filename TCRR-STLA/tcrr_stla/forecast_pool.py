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
from .traffic_index import signature_bank,traffic_topk,get_train_starts
from .traffic import global_signature
import pandas as pd


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
    use_traffic=bool(cfg.get('traffic',{}).get('enabled',True)) and cfg.get('training',{}).get('ablation','full') not in ('legacy_v23','no_traffic_retrieval')
    variant='v5' if str(cfg.get('version','')).startswith('5.') else 'v2'
    out=raw_path.with_name(raw_path.stem+(f'.tcrr_traffic_{variant}.npz' if use_traffic else f'.tcrr_original_{variant}.npz'))
    expected={'schema':25,'raw_sha256':file_sha256(raw_path),
              'backbone_sha256':file_sha256(backbone_path),
              'adapter_sha256':file_sha256(cfg['runtime']['teacher_adapter_checkpoint']) if cfg.get('runtime',{}).get('teacher_adapter_checkpoint') else None,
              'query_x':file_sha256(query_root/f'{split}_x_obs_norm.npy'),
              'query_mask':file_sha256(query_root/f'{split}_history_observed.npy'),
              'train_future':file_sha256(formal/'train_y_norm.npy'),
              'selected':int(cfg['retrieval']['final_candidates']),
              'score_weight':float(cfg['retrieval']['history_similarity_weight']),
              'split':split,
              'traffic':cfg.get('traffic',{}), 'traffic_route':use_traffic,
              'training_ablation':cfg.get('training',{}).get('ablation','full'),
              'train_history':file_sha256(formal/'train_x_full_norm.npy'),
              'train_starts':file_sha256(formal/'train_manifest.csv'),
              'query_starts':file_sha256(query_root/f'{split}_manifest.csv')}
    meta=out.with_suffix('.json')
    if out.is_file() and meta.is_file() and json.loads(meta.read_text())==expected:
        print(f'FORECAST_POOL_HIT split={split} path={out}',flush=True);return out
    base,_=load_pretrained_stla(Path(cfg['project']['root']),Path(backbone_path),device)
    teacher=MaskAwareSTLAformer(base).to(device).eval()
    if cfg.get('runtime',{}).get('teacher_adapter_checkpoint'):
        adapter=torch.load(cfg['runtime']['teacher_adapter_checkpoint'],map_location='cpu',weights_only=False)
        teacher.mask_proj.load_state_dict(adapter['mask_proj'],strict=True)
    for p in teacher.parameters():p.requires_grad_(False)
    with np.load(raw_path,allow_pickle=False) as z:
        source_ids=np.asarray(z['indices'],np.int64)
        source_scores=np.asarray(z['scores'],np.float32)
        cov=np.asarray(z['coverage'],np.float32)
    count=24 if smoke else len(source_ids)
    count=min(count,len(source_ids)); pool=source_ids.shape[1]*source_ids.shape[2]
    futures=np.load(formal/'train_y_norm.npy',mmap_mode='r')
    train_x=np.load(formal/'train_x_full_norm.npy',mmap_mode='r')
    memory_starts=get_train_starts(formal)
    query_starts=pd.read_csv(query_root/f'{split}_manifest.csv').window_start.to_numpy(np.int64)
    if use_traffic:
        memsig=signature_bank(train_x[:(256 if smoke else len(train_x))],device,
                    int(cfg.get('traffic',{}).get('signature_chunk_size',2048)))
        memory_starts_t=torch.tensor(memory_starts[:len(memsig)],device=device)
    else:memsig=None
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
            hscore=source_scores[start:stop].reshape(stop-start,pool)
            hc=np.repeat(cov[start:stop,:,None],source_ids.shape[-1],axis=2).reshape(stop-start,pool)
            if memsig is not None:
                ti,ts=traffic_topk(qs,ms,memsig,memory_starts_t,
                      torch.as_tensor(query_starts[start:stop],device=device),
                      int(cfg['experiment']['history'])+int(cfg['experiment']['horizon']),
                      min(int(cfg.get('traffic',{}).get('memory_candidate_k',12)),memsig.shape[0]//2))
                tk=ti.cpu().numpy().astype(np.int64)
                tscore=ts.cpu().numpy().astype(np.float32)
                w=float(cfg.get('traffic',{}).get('traffic_score_weight',0.25))
                ids=np.concatenate([ids,tk],axis=1)
                hscore=np.concatenate([hscore,tscore*float(cfg.get('traffic',{}).get('traffic_score_weight',.25))],axis=1)
                query_coverage=ms.float().mean(dim=(1,2)).cpu().numpy().astype(np.float32)
                hc=np.concatenate([hc,np.broadcast_to(query_coverage[:,None],tscore.shape).copy()],axis=1)
            cy=torch.as_tensor(np.array(futures[ids],copy=True),device=device,dtype=torch.float32)
            # Per-case centered forecast shape: focus on future evolution rather than scale.
            qv=pred.flatten(1);yv=cy.flatten(2)
            qz=qv-qv.mean(-1,keepdim=True);yz=yv-yv.mean(-1,keepdim=True)
            dot=(qz[:,None]*yz).sum(-1)
            sim=dot/(qz.norm(dim=-1)[:,None]*yz.norm(dim=-1)+1e-6)
            sim=torch.nan_to_num(sim,nan=-1.).clamp(-1,1)
            # Query-mask-matched reranking: all candidate traffic values are
            # compared on precisely the query-observed history coordinates.
            raw_history=torch.as_tensor(np.array(train_x[ids],copy=True),device=device,dtype=torch.float32)
            mm=ms[:,None]
            observed_difference=((raw_history-qs[:,None]).abs()*mm).sum((2,3))/mm.sum((2,3)).clamp_min(1.)
            matched_similarity=torch.exp(-observed_difference.clamp_min(0.)).clamp(0,1)
            if cfg.get('training',{}).get('ablation')!='no_mask_matched_retrieval':
                sim=0.8*sim+0.2*(2*matched_similarity-1)
            sim=sim.cpu().numpy()
            ci,cs,cc=select_unique(ids,hscore,hc,sim,k,bw)
            selected[start:stop]=ci;scores[start:stop]=cs;covers[start:stop]=cc
            if start % max(bs*100,bs)==0:print(f'FORECAST_POOL_PROGRESS split={split} {stop}/{count}',flush=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_name(out.name+'.tmp.npz')
    np.savez_compressed(tmp,indices=selected,scores=scores,coverage=covers,version=np.int32(25))
    os.replace(tmp,out);meta.write_text(json.dumps(expected,indent=2))
    print(f'FORECAST_POOL_PASS split={split} selected={k} path={out}',flush=True)
    return out
