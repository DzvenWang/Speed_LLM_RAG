from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CacheSplit:
    def __init__(self, root, split):
        r=Path(root); self.root=r; self.split=split
        def mm(name): return np.load(r/f'{split}_{name}.npy',mmap_mode='r')
        self.x=mm('x_obs_norm'); self.obs=mm('history_observed'); self.y=mm('y_raw')
        self.valid=mm('future_valid'); self.direct=mm('direct_mask'); self.support=mm('support_mask')
        self.tod=mm('tod_hist'); self.dow=mm('dow_hist'); self.meta=pd.read_csv(r/f'{split}_manifest.csv')
    def __len__(self): return len(self.x)


def cache_stats(cache_root):
    j=json.loads((Path(cache_root)/'cache_manifest.json').read_text()); first=next(iter(j['splits'].values()))
    return float(first['mean']),float(first['std'])


def deterministic_split(n, dev_n, final_n, seed):
    rng=np.random.default_rng(int(seed)); p=rng.permutation(int(n)); dev_n=min(int(dev_n),int(n)); final_n=min(int(final_n),max(int(n)-dev_n,0))
    return p[:dev_n].astype(np.int64),p[dev_n:dev_n+final_n].astype(np.int64)


class ForecastDataset(Dataset):
    """Query + Top-K train-memory support episodes. Query future is target only."""
    def __init__(self, query_split, memory_split, query_base, memory_base,
                 query_state, memory_state, query_evidence, memory_evidence,
                 topk_indices, topk_scores, indices=None):
        self.q=query_split; self.m=memory_split; self.qbase=query_base; self.mbase=memory_base
        self.qstate=query_state; self.mstate=memory_state; self.qev=query_evidence; self.mev=memory_evidence
        self.topi=topk_indices; self.tops=topk_scores
        self.indices=np.arange(len(query_split),dtype=np.int64) if indices is None else np.asarray(indices,np.int64)
    def __len__(self): return len(self.indices)
    def __getitem__(self,j):
        i=int(self.indices[j]); ci=np.asarray(self.topi[i],np.int64)
        return {
            'query_index':torch.tensor(i,dtype=torch.long),
            'query_state':torch.from_numpy(np.asarray(self.qstate[i],np.float32).copy()),
            'query_evidence':torch.from_numpy(np.asarray(self.qev[i],np.float32).copy()),
            'query_base':torch.from_numpy(np.asarray(self.qbase[i],np.float32).copy()),
            'target':torch.from_numpy(np.asarray(self.q.y[i],np.float32).copy()),
            'future_valid':torch.from_numpy(np.asarray(self.q.valid[i],bool).copy()),
            'direct_mask':torch.from_numpy(np.asarray(self.q.direct[i],bool).copy()),
            'support_mask':torch.from_numpy(np.asarray(self.q.support[i],bool).copy()),
            'candidate_indices':torch.from_numpy(ci.copy()),
            'candidate_scores':torch.from_numpy(np.asarray(self.tops[i],np.float32).copy()),
            'support_state':torch.from_numpy(np.asarray(self.mstate[ci],np.float32).copy()),
            'support_evidence':torch.from_numpy(np.asarray(self.mev[ci],np.float32).copy()),
            'support_future':torch.from_numpy(np.asarray(self.m.y[ci],np.float32).copy()),
            'support_future_valid':torch.from_numpy(np.asarray(self.m.valid[ci],bool).copy()),
            'support_base':torch.from_numpy(np.asarray(self.mbase[ci],np.float32).copy()),
        }
