from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset

class SupportSplit(Dataset):
    def __init__(self, cache: str|Path, split: str, limit: int|None=None):
        r=Path(cache); self.root=r; self.split=split
        def mm(name): return np.load(r/f'{split}_{name}.npy',mmap_mode='r')
        self.x_obs=mm('x_obs_norm'); self.x_full=mm('x_full_norm'); self.y_norm=mm('y_norm'); self.y_raw=mm('y_raw'); self.observed=mm('history_observed'); self.valid=mm('future_valid'); self.direct=mm('direct_mask'); self.tod=mm('tod_hist'); self.dow=mm('dow_hist'); self.meta=pd.read_csv(r/f'{split}_manifest.csv')
        groups=self.meta.view_group_id.astype(str).tolist() if 'view_group_id' in self.meta.columns else [f'row_{i}' for i in range(len(self.meta))]; lut={g:i for i,g in enumerate(dict.fromkeys(groups))}; self.group_ids=np.asarray([lut[g] for g in groups],np.int64)
        self.n=len(self.meta) if limit is None else min(int(limit),len(self.meta))
    def __len__(self): return self.n
    def __getitem__(self,i):
        return {'index':int(i),'x':torch.from_numpy(np.array(self.x_obs[i],np.float32,copy=True)),'x_full':torch.from_numpy(np.array(self.x_full[i],np.float32,copy=True)),'mask':torch.from_numpy(np.array(self.observed[i],np.float32,copy=True)),'y_norm':torch.from_numpy(np.array(self.y_norm[i],np.float32,copy=True)),'y_raw':torch.from_numpy(np.array(self.y_raw[i],np.float32,copy=True)),'future_valid':torch.from_numpy(np.array(self.valid[i],np.bool_,copy=True)),'direct_mask':torch.from_numpy(np.array(self.direct[i],np.bool_,copy=True)),'tod':torch.from_numpy(np.array(self.tod[i],np.int64,copy=True)),'dow':torch.from_numpy(np.array(self.dow[i],np.int64,copy=True)),'window_start':int(self.meta.iloc[i].window_start),'group_id':int(self.group_ids[i])}

class FullTrainRepository:
    def __init__(self, formal_cache: str|Path, limit: int|None=None):
        r=Path(formal_cache); self.root=r; self.x=np.load(r/'train_x_full_norm.npy',mmap_mode='r'); self.tod=np.load(r/'train_tod_hist.npy',mmap_mode='r'); self.dow=np.load(r/'train_dow_hist.npy',mmap_mode='r'); self.meta=pd.read_csv(r/'train_manifest.csv'); self.n=len(self.meta) if limit is None else min(int(limit),len(self.meta)); self.starts=np.asarray(self.meta.window_start.astype(np.int64).to_numpy()[:self.n])
    def __len__(self): return self.n
    def batch(self,start,end,device=None):
        x=torch.from_numpy(np.array(self.x[start:end],np.float32,copy=True)); tod=torch.from_numpy(np.array(self.tod[start:end],np.int64,copy=True)); dow=torch.from_numpy(np.array(self.dow[start:end],np.int64,copy=True)); m=torch.ones_like(x)
        if device is not None: x=x.to(device);tod=tod.to(device);dow=dow.to(device);m=m.to(device)
        return x,m,tod,dow
    def fetch(self,indices,device):
        ind=np.asarray(indices,dtype=np.int64); shape=ind.shape; flat=ind.reshape(-1)
        x=torch.from_numpy(np.array(self.x[flat],np.float32,copy=True)).to(device); tod=torch.from_numpy(np.array(self.tod[flat],np.int64,copy=True)).to(device); dow=torch.from_numpy(np.array(self.dow[flat],np.int64,copy=True)).to(device); m=torch.ones_like(x)
        return x.view(*shape,*x.shape[1:]),m.view(*shape,*m.shape[1:]),tod.view(*shape,*tod.shape[1:]),dow.view(*shape,*dow.shape[1:])
