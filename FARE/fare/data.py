from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader

class SupportSplit(Dataset):
    """Strict Base_SupportBench schema-v3 adapter; no synthetic fallback in formal mode."""
    def __init__(self, root, split, limit=None):
        self.root=Path(root); self.split=split
        fields=['x_obs_norm','x_full_norm','history_observed','artificial_missing','y_norm','y_raw','future_valid','direct_mask','tod_hist','dow_hist']
        self.arr={k:np.load(self.root/f'{split}_{k}.npy', mmap_mode='r') for k in fields}
        self.meta=pd.read_csv(self.root/f'{split}_manifest.csv')
        self.length=len(self.meta) if limit is None else min(int(limit),len(self.meta))
        if self.arr['x_obs_norm'].shape[0] != len(self.meta): raise ValueError('Manifest / data row mismatch')
        self.window_starts=self.meta['window_start'].to_numpy(dtype=np.int64,copy=True)
        self.num_nodes=int(self.arr['x_obs_norm'].shape[-1]);self.history=int(self.arr['x_obs_norm'].shape[1])
        self.horizon=int(self.arr['y_norm'].shape[1]);self.family_column='family' if 'family' in self.meta else None
    def __len__(self): return self.length
    def __getitem__(self,i):
        a=self.arr
        fields={'x_obs':'x_obs_norm','x_full':'x_full_norm','obs_mask':'history_observed','artificial_mask':'artificial_missing','y_norm':'y_norm','y_raw':'y_raw','future_valid':'future_valid','direct_mask':'direct_mask','tod':'tod_hist','dow':'dow_hist'}
        out={k:torch.from_numpy(np.array(a[v][i],copy=True)) for k,v in fields.items()}
        out['raw_valid']=(out['obs_mask'].bool() | out['artificial_mask'].bool())
        out['obs_mask']=out['obs_mask'].bool()
        out['artificial_mask']=out['artificial_mask'].bool()
        out['window_start']=torch.tensor(int(self.window_starts[i]),dtype=torch.long)
        out['row_id']=torch.tensor(i,dtype=torch.long)
        return out

def loader(ds,batch_size,shuffle=False,workers=0):
    return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,num_workers=workers,
                      pin_memory=torch.cuda.is_available(),persistent_workers=workers>0,
                      prefetch_factor=2 if workers>0 else None,drop_last=False)

def to_device(batch,device):
    return {k:v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}

def graph_from_cache(root):
    r=Path(root); adj=np.load(r/'adjacency.npy')
    if adj.ndim!=2 or adj.shape[0]!=adj.shape[1]: raise ValueError('Invalid adjacency')
    return torch.as_tensor(np.array(adj,np.float32,copy=True))

def directed_from_benchmark(dataset,target,root=None):
    from .config import project_root
    base=Path(root) if root is not None else project_root()
    path=base/'data'/'processed'/'traffident'/'base_supportbench'/target/dataset/'graph'/'directed_neighbors.npz'
    if not path.is_file():
        raise FileNotFoundError(f'Directed graph is required for full FARE: {path}')
    z=np.load(path,allow_pickle=False)
    return {k:torch.as_tensor(np.array(z[k],copy=True)) for k in ('upstream_ids','downstream_ids','upstream_weights','downstream_weights')}
