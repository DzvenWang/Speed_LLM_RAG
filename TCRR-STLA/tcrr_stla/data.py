from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset

class SupportSplit(Dataset):
    def __init__(self, cache: str|Path, split: str, limit: int|None=None):
        r=Path(cache); self.root=r; self.split=split
        def mm(name): return np.load(r/f'{split}_{name}.npy',mmap_mode='r')
        self.x_obs=mm('x_obs_norm'); self.y_norm=mm('y_norm'); self.y_raw=mm('y_raw')
        self.observed=mm('history_observed'); self.valid=mm('future_valid'); self.direct=mm('direct_mask')
        self.tod=mm('tod_hist'); self.dow=mm('dow_hist'); self.meta=pd.read_csv(r/f'{split}_manifest.csv')
        self.n=len(self.meta) if limit is None else min(int(limit),len(self.meta))
    def __len__(self): return self.n
    def __getitem__(self,i):
        return {
            'index':int(i),'x':torch.from_numpy(np.array(self.x_obs[i],np.float32,copy=True)),
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
    """New-format forecast-conditioned shortlist; incompatible with pre-v2.3 files."""
    def __init__(self, path):
        with np.load(path,allow_pickle=False) as z:
            if int(z['version'])!=24:raise RuntimeError('Use TCRR v1 forecast pool cache')
            self.selected_indices=np.asarray(z['indices'],np.int32)
            self.selected_scores=np.asarray(z['scores'],np.float32)
            self.selected_coverage=np.asarray(z['coverage'],np.float32)
            if not (self.selected_indices.shape==self.selected_scores.shape==self.selected_coverage.shape):
                raise ValueError('v2.3 shortlist arrays disagree')
            self.candidate_count=self.selected_indices.shape[1]
