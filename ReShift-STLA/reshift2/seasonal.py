"""Train-only future innovation reference for fully unobserved sensors.

Hourly-of-week buckets derived from LAST HISTORY timestamp only.
No query future or test complete-history data is accessed.
"""
from __future__ import annotations
import numpy as np
import torch


def train_future_reference(repository, horizon, nodes, batch_size=256):
    sums=np.zeros((168,horizon,nodes),np.float64)
    counts=np.zeros((168,horizon,nodes),np.float64)
    global_sum=np.zeros((horizon,nodes),np.float64)
    global_count=np.zeros((horizon,nodes),np.float64)
    for start in range(0,len(repository),batch_size):
        stop=min(start+batch_size,len(repository))
        y=np.array(repository.y[start:stop],dtype=np.float64,copy=True)
        valid=(np.isfinite(y) if repository.future_valid is None else
               np.array(repository.future_valid[start:stop],dtype=bool,copy=True)&np.isfinite(y))
        y=np.where(valid,y,0.0)
        ti=np.array(repository.tod[start:stop,-1],dtype=np.int64)
        di=np.array(repository.dow[start:stop,-1],dtype=np.int64)
        buck=((di%7)*24+(ti%288)//12).astype(np.int64)
        np.add.at(sums,buck,y);np.add.at(counts,buck,valid.astype(np.float64))
        global_sum+=y.sum(0);global_count+=valid.sum(0)
    fallback=global_sum/np.maximum(global_count,1)
    means=np.divide(sums,np.maximum(counts,1),out=np.zeros_like(sums))
    means=np.where(counts>=2,means,fallback[None])
    return torch.from_numpy(means.astype(np.float32))


def seasonal_innovation(future, bucket, means):
    """future [B,C,P,N], bucket [B,C,N], means [168,P,N]."""
    B,C,P,N=future.shape
    nid=torch.arange(N,device=future.device)[None,None].expand(B,C,N)
    reference=means[bucket.long().clamp(0,167),:,nid].permute(0,1,3,2)
    return torch.nan_to_num(future.float()-reference.float())
