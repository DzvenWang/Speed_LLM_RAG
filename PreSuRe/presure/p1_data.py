from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

VARIANTS=('adapter_only','shuffled_support','history_only','history_future')


class P1Dataset(Dataset):
    def __init__(self, cache_root, artifact_root, split, variant, indices=None, mean=0.0, std=1.0):
        if variant not in VARIANTS: raise ValueError(variant)
        r=Path(cache_root); a=Path(artifact_root); self.variant=variant; self.mean=float(mean); self.std=float(std)
        self.y=np.load(r/f'{split}_y_raw.npy',mmap_mode='r')
        self.valid=np.load(r/f'{split}_future_valid.npy',mmap_mode='r')
        self.direct=np.load(r/f'{split}_direct_mask.npy',mmap_mode='r')
        self.support=np.load(r/f'{split}_support_mask.npy',mmap_mode='r')
        self.base=np.load(a/f'{split}_base_pred_norm.npy',mmap_mode='r')
        self.latent=np.load(a/f'{split}_query_latent.npy',mmap_mode='r')
        self.qhist=np.load(a/f'{split}_query_history_feat.npy',mmap_mode='r')
        if variant=='shuffled_support':
            pref='random'
        else:
            pref='real'
        self.sh=np.load(a/f'{split}_{pref}_support_history.npy',mmap_mode='r')
        self.sf=np.load(a/f'{split}_{pref}_support_future.npy',mmap_mode='r')
        self.sc=np.load(a/f'{split}_{pref}_support_future_cov.npy',mmap_mode='r')
        n=len(self.y); self.ids=np.arange(n,dtype=np.int64) if indices is None else np.asarray(indices,np.int64)
    def __len__(self): return len(self.ids)
    def __getitem__(self,j):
        i=int(self.ids[j])
        sh=np.asarray(self.sh[i],np.float32); sf=np.asarray(self.sf[i],np.float32); sc=np.asarray(self.sc[i],np.float32)
        if self.variant=='adapter_only': sh=np.zeros_like(sh); sf=np.zeros_like(sf); sc=np.zeros_like(sc)
        elif self.variant=='history_only': sf=np.zeros_like(sf); sc=np.zeros_like(sc)
        y=(np.asarray(self.y[i],np.float32)-self.mean)/max(self.std,1e-8)
        return {
            'idx':torch.tensor(i,dtype=torch.long),
            'base':torch.from_numpy(np.asarray(self.base[i],np.float32).copy()),
            'latent':torch.from_numpy(np.asarray(self.latent[i],np.float32).copy()),
            'qhist':torch.from_numpy(np.asarray(self.qhist[i],np.float32).copy()),
            'shist':torch.from_numpy(sh.copy()),'sfuture':torch.from_numpy(sf.copy()),'scov':torch.from_numpy(sc.copy()),
            'target_norm':torch.from_numpy(y.copy()),
            'target_raw':torch.from_numpy(np.asarray(self.y[i],np.float32).copy()),
            'valid':torch.from_numpy(np.asarray(self.valid[i],bool).copy()),
            'direct':torch.from_numpy(np.asarray(self.direct[i],bool).copy()),
            'support':torch.from_numpy(np.asarray(self.support[i],bool).copy()),
        }
