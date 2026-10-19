"""Build sparse statistical neighbors from Train-only histories, never node-id adjacency.

This is a predictive support graph, not a claim of directed road topology.
"""
from pathlib import Path
import numpy as np
import torch


def train_only_neighbours(train_cache, max_neighbors=8, max_rows=1200):
    path=Path(train_cache)/'train_x_full_norm.npy'
    if not path.is_file():raise FileNotFoundError(f'Train-only histories required: {path}')
    x=np.load(path,mmap_mode='r')
    if x.ndim!=3 or x.shape[1]<2:raise ValueError('Expected Train history [M,H,N]')
    m,h,n=x.shape
    if n==1:return torch.zeros((1,1),dtype=torch.long)
    rows=np.linspace(0,m-1,min(m,max_rows),dtype=np.int64)
    raw=np.nan_to_num(np.array(x[rows],copy=True,dtype=np.float32))
    # Differencing suppresses shared time-of-day level effects.
    z=np.diff(raw,axis=1).reshape(-1,n).astype(np.float64)
    z-=z.mean(0,keepdims=True)
    z/=np.sqrt((z*z).mean(0,keepdims=True)+1e-6)
    correlation=(z.T@z)/max(len(z),1)
    np.fill_diagonal(correlation,0)
    k=min(max(int(max_neighbors),1),n-1)
    # Stable sort independent of hardware, deterministic for tied scores.
    neighbors=np.argsort(-np.abs(correlation),axis=1,kind='stable')[:,:k].copy()
    return torch.from_numpy(neighbors.astype(np.int64))
