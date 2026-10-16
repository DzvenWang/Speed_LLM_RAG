"""Independent train-only traffic-regime candidate discovery for TCRR.

The similarity index is built from TRAIN HISTORY, never train future. It is
queried using visible masked query HISTORY. Memory windows overlapping the
query's complete history+horizon interval are excluded before top-k.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .traffic import global_signature


def signature_bank(full_history,device,chunk=1024):
    chunks=[]
    with torch.no_grad():
        for i in range(0,len(full_history),int(chunk)):
            x=torch.tensor(np.array(full_history[i:i+chunk],copy=True),device=device,dtype=torch.float32)
            chunks.append(global_signature(x,torch.ones_like(x)).cpu())
    return torch.cat(chunks,0).to(device)


def traffic_topk(query_history,query_mask,memory_signatures,starts,query_starts,span,k):
    with torch.no_grad(),torch.autocast(device_type=query_history.device.type,enabled=False):
        z=global_signature(query_history.float(),query_mask.float())
        memory=memory_signatures.float()
        # Train-only index standardization prevents a high-variance flow level
        # from overwhelming the trend, missing-age and state reliability axes.
        mu=memory.mean(0,keepdim=True);sd=memory.std(0,keepdim=True).clamp_min(0.05)
        a=torch.nn.functional.normalize((z-mu)/sd,dim=-1)
        m=torch.nn.functional.normalize((memory-mu)/sd,dim=-1)
        sim=a@m.T
        qstart=query_starts.to(sim.device).long().reshape(-1,1)
        mstart=starts.to(sim.device).long().reshape(1,-1)
        legal=(mstart>=qstart+span)|(qstart>=mstart+span)
        sim=sim.masked_fill(~legal,-torch.inf)
        if not (torch.isfinite(sim).sum(-1)>=int(k)).all():
            raise RuntimeError('Insufficient temporally legal traffic candidates')
        values,ids=sim.topk(int(k),dim=-1)
        return ids,values


def get_train_starts(cache):
    return pd.read_csv(Path(cache)/'train_manifest.csv').window_start.to_numpy(np.int64)
