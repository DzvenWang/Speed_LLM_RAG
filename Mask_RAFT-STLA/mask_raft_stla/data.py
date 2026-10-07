from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset

class SupportSplit(Dataset):
    def __init__(self, cache: str|Path, split: str, limit: int|None=None):
        r=Path(cache); self.root=r; self.split=split
        def mm(name): return np.load(r/f'{split}_{name}.npy',mmap_mode='r')
        self.x_obs=mm('x_obs_norm'); self.x_full=mm('x_full_norm'); self.y_norm=mm('y_norm'); self.y_raw=mm('y_raw')
        self.observed=mm('history_observed'); self.valid=mm('future_valid'); self.direct=mm('direct_mask')
        self.tod=mm('tod_hist'); self.dow=mm('dow_hist'); self.meta=pd.read_csv(r/f'{split}_manifest.csv')
        self.n=len(self.meta) if limit is None else min(int(limit),len(self.meta))
    def __len__(self): return self.n
    def __getitem__(self,i):
        return {
            'index':int(i),'x':torch.from_numpy(np.array(self.x_obs[i],np.float32,copy=True)),
            'x_full':torch.from_numpy(np.array(self.x_full[i],np.float32,copy=True)),
            'mask':torch.from_numpy(np.array(self.observed[i],np.float32,copy=True)),
            'y_norm':torch.from_numpy(np.array(self.y_norm[i],np.float32,copy=True)),
            'y_raw':torch.from_numpy(np.array(self.y_raw[i],np.float32,copy=True)),
            'future_valid':torch.from_numpy(np.array(self.valid[i],np.bool_,copy=True)),
            'direct_mask':torch.from_numpy(np.array(self.direct[i],np.bool_,copy=True)),
            'tod':torch.from_numpy(np.array(self.tod[i],np.int64,copy=True)),
            'dow':torch.from_numpy(np.array(self.dow[i],np.int64,copy=True)),
            'window_start':int(self.meta.iloc[i].window_start),
        }

class RetrievalCache:
    def __init__(self,path: str|Path):
        z=np.load(path,mmap_mode='r',allow_pickle=False)
        self.indices=z['indices']; self.scores=z['scores']; self.coverage=z['coverage']
        self.periods=[int(x) for x in z['periods'].tolist()]; self.top_k=int(self.indices.shape[-1])
    def batch_evidence(self, batch_indices, memory_y, temperature: float):
        idx=np.asarray(batch_indices,dtype=np.int64)
        top=np.asarray(self.indices[idx],np.int64)  # B,G,K
        scores=np.asarray(self.scores[idx],np.float32)
        cov=np.asarray(self.coverage[idx],np.float32)
        # Gather only K selected full-train futures per period, then reduce on CPU.
        gathered=np.asarray(memory_y[top],np.float32) # B,G,K,P,N
        logits=scores/max(float(temperature),1e-6); logits=logits-logits.max(axis=-1,keepdims=True)
        w=np.exp(logits); w/=np.maximum(w.sum(axis=-1,keepdims=True),1e-12)
        ev=(gathered*w[...,None,None]).sum(axis=2).astype(np.float32)
        p=np.clip(w,1e-12,1.0); ent=-(p*np.log(p)).sum(axis=-1)/np.log(max(self.top_k,2))
        top1=scores[...,0]; margin=top1-scores[...,1] if self.top_k>1 else np.zeros_like(top1)
        conf=np.stack([top1,margin,1.0-ent,cov],axis=-1).astype(np.float32)
        return torch.from_numpy(ev), torch.from_numpy(conf)
