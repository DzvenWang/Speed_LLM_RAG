"""Strict Base_SupportBench schema-v3 bridge. No synthetic fallback in production."""
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset,DataLoader,Subset

class SupportDataset(Dataset):
    fields=('x_obs_norm','x_full_norm','history_observed','artificial_missing','y_norm','y_raw','future_valid','direct_mask','tod_hist','dow_hist')
    def __init__(self,path,split,limit=None,strategy='head'):
        self.path=Path(path);self.split=split
        self.arr={k:np.load(self.path/f'{split}_{k}.npy',mmap_mode='r') for k in self.fields}
        self.meta=pd.read_csv(self.path/f'{split}_manifest.csv')
        if len(self.meta)!=len(self.arr['y_norm']):raise RuntimeError('SupportBench manifest mismatch')
        if 'window_start' not in self.meta:raise RuntimeError('Missing window_start in manifest')
        ix=np.arange(len(self.meta),dtype=np.int64)
        if limit is not None and int(limit)<len(ix):
            ix=ix[np.linspace(0,len(ix)-1,int(limit),dtype=np.int64)] if strategy=='spread' else ix[:int(limit)]
        self.ix=ix
        self._window_starts=self.meta.window_start.to_numpy(dtype=np.int64,copy=True)
        self.n=int(self.arr['x_obs_norm'].shape[-1]);self.h=int(self.arr['x_obs_norm'].shape[-2]);self.p=int(self.arr['y_norm'].shape[-2])
    def __len__(self):return len(self.ix)
    def __getitem__(self,i):
        j=int(self.ix[i]);a=self.arr
        def t(k,dtype=None):
            v=torch.from_numpy(np.array(a[k][j],copy=True))
            return v.to(dtype) if dtype is not None else v
        ob=t('history_observed',torch.bool);missing=t('artificial_missing',torch.bool)
        return {'x':t('x_obs_norm',torch.float32),'x_full':t('x_full_norm',torch.float32),'mask':ob,
                'raw_valid':ob|missing,'target':t('y_norm',torch.float32),'target_raw':t('y_raw',torch.float32),
                'future_valid':t('future_valid',torch.bool),'direct_mask':t('direct_mask',torch.bool),
                'tod':t('tod_hist',torch.long),'dow':t('dow_hist',torch.long),
                'origin':torch.tensor(int(self._window_starts[j])+self.h-1,dtype=torch.long),
                'row_id':torch.tensor(j,dtype=torch.long)}

def make_loader(ds,batch_size,shuffle=False,workers=0):
    return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,num_workers=workers,
                      pin_memory=torch.cuda.is_available(),drop_last=False,
                      persistent_workers=bool(workers),prefetch_factor=2 if workers else None)

def device_batch(b,device):return {k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in b.items()}

def graph_files(path):
    p=Path(path);a=np.load(p/'adjacency.npy');n=a.shape[0]
    if a.shape!=(n,n):raise ValueError('Adjacency must be square')
    graph={'adj':torch.as_tensor(np.array(a,dtype=np.float32,copy=True))}
    graph['metadata_source']='undirected_adjacency_only'
    return graph

def directed_metadata(dataset,target,root):
    p=Path(root)/'data'/'processed'/'traffident'/'base_supportbench'/target/dataset/'graph'/'directed_neighbors.npz'
    if not p.exists():return None
    z=np.load(p,allow_pickle=False)
    req=('upstream_ids','downstream_ids','upstream_weights','downstream_weights')
    if any(k not in z for k in req):raise ValueError(f'Incomplete directional data at {p}')
    return {k:torch.as_tensor(np.array(z[k],copy=True)) for k in req}
