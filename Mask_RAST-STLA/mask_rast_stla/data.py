from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset

class SupportSplit(Dataset):
    def __init__(self, cache: str|Path, split: str, limit: int|None=None):
        r=Path(cache); self.root=r; self.split=split
        def mm(name): return np.load(r/f'{split}_{name}.npy',mmap_mode='r')
        self.x_obs=mm('x_obs_norm'); self.y_norm=mm('y_norm'); self.y_raw=mm('y_raw'); self.observed=mm('history_observed'); self.valid=mm('future_valid'); self.direct=mm('direct_mask'); self.tod=mm('tod_hist'); self.dow=mm('dow_hist'); self.meta=pd.read_csv(r/f'{split}_manifest.csv')
        groups=self.meta.view_group_id.astype(str).tolist() if 'view_group_id' in self.meta.columns else [f'row_{i}' for i in range(len(self.meta))]; lut={g:i for i,g in enumerate(dict.fromkeys(groups))}; self.group_ids=np.asarray([lut[g] for g in groups],np.int64)
        self.n=len(self.meta) if limit is None else min(int(limit),len(self.meta))
    def __len__(self): return self.n
    def __getitem__(self,i):
        return {'index':int(i),'x':torch.from_numpy(np.array(self.x_obs[i],np.float32,copy=True)),'mask':torch.from_numpy(np.array(self.observed[i],np.float32,copy=True)),'y_norm':torch.from_numpy(np.array(self.y_norm[i],np.float32,copy=True)),'y_raw':torch.from_numpy(np.array(self.y_raw[i],np.float32,copy=True)),'future_valid':torch.from_numpy(np.array(self.valid[i],np.bool_,copy=True)),'direct_mask':torch.from_numpy(np.array(self.direct[i],np.bool_,copy=True)),'tod':torch.from_numpy(np.array(self.tod[i],np.int64,copy=True)),'dow':torch.from_numpy(np.array(self.dow[i],np.int64,copy=True)),'window_start':int(self.meta.iloc[i].window_start),'group_id':int(self.group_ids[i])}

class FullTrainRepository:
    def __init__(self, formal_cache: str|Path, limit: int|None=None):
        r=Path(formal_cache); self.root=r; self.x=np.load(r/'train_x_full_norm.npy',mmap_mode='r'); self.y=np.load(r/'train_y_norm.npy',mmap_mode='r'); self.future_valid=np.load(r/'train_future_valid.npy',mmap_mode='r') if (r/'train_future_valid.npy').is_file() else None; self.tod=np.load(r/'train_tod_hist.npy',mmap_mode='r'); self.dow=np.load(r/'train_dow_hist.npy',mmap_mode='r'); self.meta=pd.read_csv(r/'train_manifest.csv'); self.n=len(self.meta) if limit is None else min(int(limit),len(self.meta)); self.starts=np.asarray(self.meta.window_start.astype(np.int64).to_numpy()[:self.n])
        self.x_device=None; self.tod_device=None; self.dow_device=None; self._device=None
    def __len__(self): return self.n
    def preload(self,device):
        device=torch.device(device)
        if device.type!='cuda': return False
        if self._device==str(device): return True
        try:
            self.x_device=torch.from_numpy(np.array(self.x[:self.n],np.float32,copy=True)).to(device)
            self.tod_device=torch.from_numpy(np.array(self.tod[:self.n],np.int64,copy=True)).to(device)
            self.dow_device=torch.from_numpy(np.array(self.dow[:self.n],np.int64,copy=True)).to(device)
            self._device=str(device)
            gb=(self.x_device.numel()*self.x_device.element_size()+self.tod_device.numel()*self.tod_device.element_size()+self.dow_device.numel()*self.dow_device.element_size())/1024**3
            print(f'REPOSITORY_GPU rows={self.n} size_gb={gb:.3f}',flush=True)
            return True
        except RuntimeError as e:
            if 'out of memory' not in str(e).lower(): raise
            self.x_device=self.tod_device=self.dow_device=None; self._device=None; torch.cuda.empty_cache()
            print(f'REPOSITORY_GPU_FALLBACK reason={str(e).splitlines()[0]}',flush=True)
            return False
    def batch(self,start,end,device=None):
        if device is not None and self._device==str(torch.device(device)):
            x=self.x_device[start:end]; tod=self.tod_device[start:end]; dow=self.dow_device[start:end]; m=torch.ones_like(x); return x,m,tod,dow
        x=torch.from_numpy(np.array(self.x[start:end],np.float32,copy=True)); tod=torch.from_numpy(np.array(self.tod[start:end],np.int64,copy=True)); dow=torch.from_numpy(np.array(self.dow[start:end],np.int64,copy=True)); m=torch.ones_like(x)
        if device is not None: x=x.to(device,non_blocking=True);tod=tod.to(device,non_blocking=True);dow=dow.to(device,non_blocking=True);m=m.to(device,non_blocking=True)
        return x,m,tod,dow
    def fetch(self,indices,device):
        d=torch.device(device)
        if torch.is_tensor(indices):
            shape=tuple(indices.shape)
            if self._device==str(d):
                tind=indices.reshape(-1).to(device=d,dtype=torch.long,non_blocking=True); x=self.x_device[tind]; tod=self.tod_device[tind]; dow=self.dow_device[tind]; m=torch.ones_like(x)
            else:
                flat=indices.detach().cpu().numpy().astype(np.int64,copy=False).reshape(-1); x=torch.from_numpy(np.array(self.x[flat],np.float32,copy=True)).to(d,non_blocking=True); tod=torch.from_numpy(np.array(self.tod[flat],np.int64,copy=True)).to(d,non_blocking=True); dow=torch.from_numpy(np.array(self.dow[flat],np.int64,copy=True)).to(d,non_blocking=True); m=torch.ones_like(x)
        else:
            ind=np.asarray(indices,dtype=np.int64); shape=ind.shape; flat=ind.reshape(-1)
            if self._device==str(d):
                tind=torch.as_tensor(flat,dtype=torch.long,device=d); x=self.x_device[tind]; tod=self.tod_device[tind]; dow=self.dow_device[tind]; m=torch.ones_like(x)
            else:
                x=torch.from_numpy(np.array(self.x[flat],np.float32,copy=True)).to(d,non_blocking=True); tod=torch.from_numpy(np.array(self.tod[flat],np.int64,copy=True)).to(d,non_blocking=True); dow=torch.from_numpy(np.array(self.dow[flat],np.int64,copy=True)).to(d,non_blocking=True); m=torch.ones_like(x)
        return x.view(*shape,*x.shape[1:]),m.view(*shape,*m.shape[1:]),tod.view(*shape,*tod.shape[1:]),dow.view(*shape,*dow.shape[1:])

    def fetch_future(self,indices,device):
        shape=tuple(indices.shape)
        flat=indices.detach().cpu().numpy().astype(np.int64).reshape(-1)
        if (flat<0).any() or (flat>=self.n).any():raise IndexError('Future Memory must use train-only indices')
        return torch.from_numpy(np.array(self.y[flat],np.float32,copy=True)).to(device).reshape(*shape,*self.y.shape[1:])

    def fetch_future_valid(self,indices,device):
        if self.future_valid is None:
            return torch.ones((*indices.shape,*self.y.shape[1:]),device=device,dtype=torch.bool)
        shape=tuple(indices.shape)
        flat=indices.detach().cpu().numpy().astype(np.int64).reshape(-1)
        return torch.from_numpy(np.array(self.future_valid[flat],copy=True)).to(device).reshape(*shape,*self.y.shape[1:]).bool()
