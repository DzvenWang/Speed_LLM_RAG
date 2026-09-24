from __future__ import annotations
import numpy as np


def ranking_metrics(pred_score,distance,top_k=8):
    p=np.asarray(pred_score,np.float64); d=np.asarray(distance,np.float64); k=min(int(top_k),p.shape[1]); pred=np.argsort(-p,axis=1)[:,:k]; best=np.argmin(d,axis=1)
    recall=float(np.mean((pred==best[:,None]).any(1))); order=np.argsort(d,axis=1); rank=np.empty_like(order); rows=np.arange(len(d))[:,None]; rank[rows,order]=np.arange(d.shape[1])[None,:]; rel=1.0/(1.0+rank.astype(float)); disc=1.0/np.log2(np.arange(k)+2.0); dcg=(rel[rows,pred]*disc[None]).sum(1); ideal=np.sort(rel,axis=1)[:,::-1][:,:k]; ndcg=float(np.mean(dcg/np.maximum((ideal*disc[None]).sum(1),1e-12)))
    pb=np.take_along_axis(d,pred,1).min(1); ob=d.min(1); span=np.maximum(d.max(1)-d.min(1),1e-8); regret=float(np.mean((pb-ob)/span))
    return {f'recall_at_{k}':recall,f'ndcg_at_{k}':ndcg,f'normalized_future_regret_at_{k}':regret}
