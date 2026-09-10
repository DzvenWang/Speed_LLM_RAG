#!/usr/bin/env python3
"""Build the large candidate mining pool used to train Innovation 1.

This is deliberately separate from formal retrieval evaluation.  Training
mining combines hard temporal/spatial/propagation candidates with temporally
diverse Train-memory negatives, so the learned retriever is not capped by the
control retriever's evaluation Top-64 pool.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from supportbench.config import load_config
from supportbench.dataset import SupportBench
from supportbench.features import multi_view_features, l2_normalize_rows
from supportbench.memory import candidate_allowed
from supportbench.utils import ensure_dir, log, uniform_temporal_subsample


def norm_speed(b: SupportBench, x: np.ndarray) -> np.ndarray:
    st=b.stats['channels'][b.manifest['primary_channel']]
    return (np.asarray(x,np.float32)-float(st['mean']))/max(float(st['std']),1e-6)


def select_queries(b,args):
    if args.views_file:
        d=pd.read_csv(args.views_file); col=args.view_id_column
        if col not in d.columns: raise KeyError(f'{col} missing from {args.views_file}')
        ids=list(map(str,d[col])); rows=b.views[b.views.view_id.astype(str).isin(set(ids))].copy()
        order={v:i for i,v in enumerate(ids)}; rows['_order']=rows.view_id.astype(str).map(order); rows=rows.sort_values('_order').drop(columns='_order').reset_index(drop=True)
        if len(rows)!=len(ids): raise RuntimeError(f'Protocol requested {len(ids)} views but benchmark resolved {len(rows)}')
        return rows
    df=b.views_for(args.split,family=args.family,severity=args.severity,distribution=args.distribution)
    if args.instance is not None: df=df[df.instance.astype(int)==int(args.instance)].reset_index(drop=True)
    return df


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default=None); ap.add_argument('--subgraph',default='TraffiDent_Net_06')
    ap.add_argument('--views-file',default=None); ap.add_argument('--view-id-column',default='view_id')
    ap.add_argument('--split',default='train',choices=['train','val','test']); ap.add_argument('--family',default='regional_outage'); ap.add_argument('--severity',type=float,default=0.40); ap.add_argument('--distribution',default='id'); ap.add_argument('--instance',type=int,default=0)
    ap.add_argument('--memory-budget',default='full',choices=['full','development']); ap.add_argument('--max-queries',type=int,default=0)
    ap.add_argument('--k',type=int,default=0); ap.add_argument('--per-view-k',type=int,default=0); ap.add_argument('--diverse-k',type=int,default=-1); ap.add_argument('--output',default=None)
    args=ap.parse_args(); cfg=load_config(args.config); b=SupportBench(args.subgraph,config=args.config)
    K=int(args.k or cfg['retrieval'].get('training_mining_k',256)); per_view=int(args.per_view_k or cfg['retrieval'].get('training_mining_per_view_k',96)); diverse_k=int(cfg['retrieval'].get('training_mining_diverse_k',64) if args.diverse_k<0 else args.diverse_k)
    diverse_k=max(0,min(diverse_k,K)); hard_quota=K-diverse_k
    views=list(cfg['retrieval']['views']); weights={v:float(cfg['retrieval']['view_weights'][v]) for v in views}; sw=sum(weights.values()); weights={k:v/max(sw,1e-12) for k,v in weights.items()}
    df=select_queries(b,args)
    if args.max_queries>0 and len(df)>args.max_queries:
        keep=uniform_temporal_subsample(np.arange(len(df)),args.max_queries); df=df.iloc[keep].reset_index(drop=True)
    if df.empty: raise RuntimeError('No query views selected')

    memory_ids=b.memory_episode_ids_for_budget(args.memory_budget); M=len(memory_ids)
    if not M: raise RuntimeError(f'No memory episodes for budget={args.memory_budget}')
    cache=Path(cfg['project']['derived_root'])/args.subgraph/'retrieval_cache'/f'multiview_memory_{args.memory_budget}.npz'
    mem=None
    if cache.exists():
        z=np.load(cache,allow_pickle=False)
        if np.array_equal(z['memory_episode_id'],memory_ids) and all(v in z.files for v in views):
            mem={v:np.asarray(z[v],np.float32) for v in views}; log(f'loaded memory feature cache {cache}')
    if mem is None:
        parts={v:[] for v in views}; log(f'building mining memory features: episodes={M:,}')
        for ii,eid in enumerate(memory_ids):
            ep=b.memory_episode(int(eid)); x=norm_speed(b,ep['history_speed']); m=np.ones_like(x,bool); ff=multi_view_features(x,m,b.dependency_ids,b.dependency_weights)
            for v in views: parts[v].append(ff[v])
            if (ii+1)%2000==0: log(f'memory features {ii+1:,}/{M:,}')
        mem={v:l2_normalize_rows(np.asarray(parts[v],np.float32)) for v in views}; ensure_dir(cache.parent); np.savez_compressed(cache,memory_episode_id=memory_ids,**mem); log(f'wrote {cache}')
    search_k=min(M,max(per_view*4,hard_quota*2,128))
    nn={v:NearestNeighbors(n_neighbors=search_k,metric='cosine',algorithm='brute',n_jobs=-1).fit(mem[v]) for v in views}
    local={int(e):i for i,e in enumerate(memory_ids)}; strict=bool(cfg['memory']['strict_temporal_exclusion']); starts=np.asarray([b.memory_starts[int(e)] for e in memory_ids],np.int64)
    bit={v:(1<<i) for i,v in enumerate(views)}; diverse_bit=1<<len(views)

    out_ids=np.full((len(df),K),-1,np.int64); combined=np.full((len(df),K),-np.inf,np.float32); source=np.zeros((len(df),K),np.uint8); counts=np.zeros(len(df),np.int32)
    per_scores={v:np.full((len(df),K),-np.inf,np.float32) for v in views}; view_ids=[]; window_starts=[]
    for qi,row in df.iterrows():
        sample=b.sample(row.to_dict()); x=norm_speed(b,sample.history_full[...,b.speed_idx]); m=sample.observation_mask[...,b.speed_idx]
        qf=multi_view_features(x,m,b.dependency_ids,b.dependency_weights); q={v:l2_normalize_rows(qf[v][None,:])[0] for v in views}
        union={}
        for v in views:
            _,pos=nn[v].kneighbors(q[v][None,:],return_distance=True); accepted=0
            for lp in pos[0]:
                eid=int(memory_ids[int(lp)]); cstart=int(b.memory_starts[eid])
                if str(row['split'])=='train' and not candidate_allowed(int(row.window_start),cstart,b.H,b.P,strict): continue
                union[eid]=union.get(eid,0)|bit[v]; accepted+=1
                if accepted>=per_view: break
        scored=[]
        for eid,sm in union.items():
            lp=local[eid]; ss={v:float(np.dot(q[v],mem[v][lp])) for v in views}; cs=sum(weights[v]*ss[v] for v in views); scored.append((cs,eid,ss,sm))
        scored.sort(key=lambda z:(-z[0],z[1])); selected=scored[:hard_quota]
        selected_ids={int(z[1]) for z in selected}

        # Temporally diverse negatives from the legal memory prefix/full archive.
        legal=[]
        for eid,cstart in zip(memory_ids,starts):
            eid=int(eid)
            if eid in selected_ids: continue
            if str(row['split'])=='train' and not candidate_allowed(int(row.window_start),int(cstart),b.H,b.P,strict): continue
            legal.append(eid)
        if diverse_k>0 and legal:
            legal=np.asarray(legal,np.int64); legal=legal[np.argsort([b.memory_starts[int(e)] for e in legal])]
            diverse=uniform_temporal_subsample(legal,min(diverse_k,len(legal)))
            for eid0 in diverse:
                eid=int(eid0); lp=local[eid]; ss={v:float(np.dot(q[v],mem[v][lp])) for v in views}; cs=sum(weights[v]*ss[v] for v in views)
                selected.append((cs,eid,ss,diverse_bit))
        selected=selected[:K]
        for j,(cs,eid,ss,sm) in enumerate(selected):
            out_ids[qi,j]=eid; combined[qi,j]=cs; source[qi,j]=sm
            for v in views: per_scores[v][qi,j]=ss[v]
        counts[qi]=len(selected); view_ids.append(str(row.view_id)); window_starts.append(int(row.window_start))
        if (qi+1)%250==0: log(f'mining queries {qi+1:,}/{len(df):,}; mean candidates={counts[:qi+1].mean():.1f}')

    out=Path(args.output) if args.output else Path(cfg['project']['derived_root'])/args.subgraph/'candidate_pools'/f'training_mining_{args.memory_budget}_k{K}.npz'; ensure_dir(out.parent)
    payload={'view_id':np.asarray(view_ids),'window_start':np.asarray(window_starts,np.int64),'candidate_episode_id':out_ids,'retrieval_score':combined,'candidate_source_mask':source,'valid_candidate_count':counts,'memory_budget':np.asarray(args.memory_budget),'retrieval_mode':np.asarray('multi_view_plus_temporal_diverse_mining'),'retrieval_views':np.asarray(views),'hard_candidate_quota':np.int32(hard_quota),'diverse_candidate_quota':np.int32(diverse_k)}
    for v in views: payload[f'{v}_score']=per_scores[v]
    np.savez_compressed(out,**payload); log(f'wrote {out}; queries={len(df):,}; K={K}; hard={hard_quota}; diverse={diverse_k}; memory={M:,}')
if __name__=='__main__': main()
