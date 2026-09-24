from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _select_channel(a,channel,name):
    if a.ndim==2: return a
    if a.ndim==3:
        c=int(channel)
        if c<0 or c>=a.shape[2]: raise RuntimeError(f'{name} channel={c} outside shape={a.shape}')
        return a[:,:,c]
    raise RuntimeError(f'{name} must be [T,N] or [T,N,C], got {a.shape}')


class CanonicalHistorySource:
    def __init__(self,cfg,num_nodes):
        root=Path(cfg['project']['benchmark_root'])/cfg['experiment']['subgraph']
        cc=cfg['canonical_history']
        self.speed_path=root/cc['speed_file']; self.valid_path=root/cc['valid_file']
        if not self.speed_path.exists(): raise FileNotFoundError(self.speed_path)
        if not self.valid_path.exists(): raise FileNotFoundError(self.valid_path)
        raw_speed=np.load(self.speed_path,mmap_mode='r'); raw_valid=np.load(self.valid_path,mmap_mode='r')
        self.speed=_select_channel(raw_speed,cc['speed_channel'],'traffic_2024').astype(np.float32,copy=False)
        self.valid=_select_channel(raw_valid,cc['valid_channel'],'raw_valid_mask').astype(bool,copy=False)
        if self.speed.shape!=self.valid.shape: raise RuntimeError(f'Canonical speed/valid mismatch {self.speed.shape} vs {self.valid.shape}')
        if self.speed.shape[1]!=int(num_nodes): raise RuntimeError(f'Canonical N={self.speed.shape[1]} != cache N={num_nodes}')
        self.offset=int(cc.get('history_offset',0)); self.H=int(cfg['experiment']['history'])
    def window(self,start):
        s=int(start)+self.offset; e=s+self.H
        if s<0 or e>self.speed.shape[0]: raise IndexError(f'Canonical history [{s},{e}) outside T={self.speed.shape[0]}')
        return np.asarray(self.speed[s:e],np.float32),np.asarray(self.valid[s:e],bool)


class OracleValDataset(Dataset):
    def __init__(self,cfg,cache_root,mean,std):
        r=Path(cache_root); self.cfg=cfg; self.mean=float(mean); self.std=float(std)
        self.x=np.load(r/'val_x_obs_norm.npy',mmap_mode='r'); self.obs=np.load(r/'val_history_observed.npy',mmap_mode='r')
        self.y=np.load(r/'val_y_raw.npy',mmap_mode='r'); self.valid_f=np.load(r/'val_future_valid.npy',mmap_mode='r')
        self.direct=np.load(r/'val_direct_mask.npy',mmap_mode='r'); self.support=np.load(r/'val_support_mask.npy',mmap_mode='r')
        self.tod=np.load(r/'val_tod_hist.npy',mmap_mode='r'); self.dow=np.load(r/'val_dow_hist.npy',mmap_mode='r')
        self.meta=pd.read_csv(r/'val_manifest.csv'); col=cfg['canonical_history']['manifest_start_column']
        if col not in self.meta.columns: raise RuntimeError(f'Manifest missing {col}; columns={list(self.meta.columns)}')
        self.starts=np.asarray(self.meta[col],np.int64)
        self.canon=CanonicalHistorySource(cfg,self.x.shape[2])
    def __len__(self): return len(self.x)
    def __getitem__(self,i):
        xo=np.asarray(self.x[i],np.float32).copy(); ho=np.asarray(self.obs[i],bool)
        raw,rv=self.canon.window(self.starts[i])
        if np.any(rv & ~np.isfinite(raw)): raise RuntimeError(f'raw_valid marks nonfinite canonical speed at query={i}')
        cn=(raw-self.mean)/self.std
        rest=rv & ~ho
        xc=xo.copy(); xc[rest]=cn[rest]
        miss_ratio=float(rest.sum()/max(rv.sum(),1))
        return {
            'index':torch.tensor(i,dtype=torch.long),'x_obs':torch.from_numpy(xo),'x_complete':torch.from_numpy(xc),
            'restorable':torch.from_numpy(rest.copy()),'raw_valid_hist':torch.from_numpy(rv.copy()),
            'y_raw':torch.from_numpy(np.asarray(self.y[i],np.float32).copy()),
            'future_valid':torch.from_numpy(np.asarray(self.valid_f[i],bool).copy()),
            'direct_mask':torch.from_numpy(np.asarray(self.direct[i],bool).copy()),
            'support_mask':torch.from_numpy(np.asarray(self.support[i],bool).copy()),
            'tod_hist':torch.from_numpy(np.asarray(self.tod[i],np.int64).copy()),
            'dow_hist':torch.from_numpy(np.asarray(self.dow[i],np.int64).copy()),
            'missing_ratio':torch.tensor(miss_ratio,dtype=torch.float32),
        }


class OracleCachedDataset(Dataset):
    """Train/Validation cache view for observation-aware PatchSTG experiments.

    ``artificial_missing`` is the only reconstruction/restoration target.  It
    identifies benchmark-hidden cells whose canonical values are available;
    naturally invalid historical cells are never used as supervision.
    """
    def __init__(self,cache_root,split):
        r=Path(cache_root); self.split=str(split)
        required=[
            'x_obs_norm','x_full_norm','history_observed','artificial_missing',
            'y_raw','y_norm','future_valid','direct_mask','support_mask',
            'tod_hist','dow_hist',
        ]
        for name in required:
            p=r/f'{self.split}_{name}.npy'
            if not p.exists(): raise FileNotFoundError(p)
            setattr(self,name,np.load(p,mmap_mode='r'))
        self.meta=pd.read_csv(r/f'{self.split}_manifest.csv')
        n=len(self.x_obs_norm)
        for name in required[1:]:
            if len(getattr(self,name))!=n:
                raise RuntimeError(f'{self.split}_{name} rows={len(getattr(self,name))} != {n}')
    def __len__(self): return len(self.x_obs_norm)
    def __getitem__(self,i):
        xo=np.asarray(self.x_obs_norm[i],np.float32).copy()
        xf=np.asarray(self.x_full_norm[i],np.float32).copy()
        obs=np.asarray(self.history_observed[i],bool).copy()
        artificial=np.asarray(self.artificial_missing[i],bool).copy()
        restorable=artificial & np.isfinite(xf)
        return {
            'index':torch.tensor(i,dtype=torch.long),
            'x_obs':torch.from_numpy(xo),'x_complete':torch.from_numpy(xf),
            'history_observed':torch.from_numpy(obs),
            'restorable':torch.from_numpy(restorable),
            'y_raw':torch.from_numpy(np.asarray(self.y_raw[i],np.float32).copy()),
            'y_norm':torch.from_numpy(np.asarray(self.y_norm[i],np.float32).copy()),
            'future_valid':torch.from_numpy(np.asarray(self.future_valid[i],bool).copy()),
            'direct_mask':torch.from_numpy(np.asarray(self.direct_mask[i],bool).copy()),
            'support_mask':torch.from_numpy(np.asarray(self.support_mask[i],bool).copy()),
            'tod_hist':torch.from_numpy(np.asarray(self.tod_hist[i],np.int64).copy()),
            'dow_hist':torch.from_numpy(np.asarray(self.dow_hist[i],np.int64).copy()),
        }


def audit_canonical_source(cfg,cache_root,mean,std,max_queries=None):
    ds=OracleValDataset(cfg,cache_root,mean,std); n=min(len(ds),int(max_queries or cfg['canonical_history']['audit_queries']))
    abs_sum=0.0; count=0; max_abs=0.0; obs_raw_invalid=0; obs_count=0; rest=0; rv_count=0; replacement_diff=0.0
    for i in range(n):
        xo=np.asarray(ds.x[i],np.float32); ho=np.asarray(ds.obs[i],bool); raw,rv=ds.canon.window(ds.starts[i]); cn=(raw-mean)/std
        m=ho & rv & np.isfinite(cn)
        if np.any(m):
            d=np.abs(xo[m]-cn[m]); abs_sum+=float(d.sum()); count+=int(d.size); max_abs=max(max_abs,float(d.max()))
        obs_raw_invalid+=int((ho & ~rv).sum()); obs_count+=int(ho.sum()); rest+=int((rv & ~ho).sum()); rv_count+=int(rv.sum())
        x_masked=xo.copy(); x_masked[rv & ~ho]=cn[rv & ~ho]
        x_all=xo.copy(); x_all[rv]=cn[rv]
        replacement_diff=max(replacement_diff,float(np.max(np.abs(x_masked-x_all))))
    return {
        'speed_path':str(ds.canon.speed_path),'valid_path':str(ds.canon.valid_path),'speed_shape':list(ds.canon.speed.shape),
        'manifest_start_column':cfg['canonical_history']['manifest_start_column'],'history_offset':int(ds.canon.offset),
        'audited_queries':n,'observed_match_mae':abs_sum/max(count,1),'observed_match_max_abs':max_abs,
        'observed_on_raw_invalid_fraction':obs_raw_invalid/max(obs_count,1),'restorable_fraction_of_raw_valid':rest/max(rv_count,1),
        'masked_only_vs_all_raw_valid_max_abs':replacement_diff,
    }
