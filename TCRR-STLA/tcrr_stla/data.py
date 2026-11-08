from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset

class SupportSplit(Dataset):
    def __init__(self, cache: str|Path, split: str, limit: int|None=None, cf_training: bool=False):
        r=Path(cache); self.root=r; self.split=split
        def mm(name): return np.load(r/f'{split}_{name}.npy',mmap_mode='r')
        self.x_obs=mm('x_obs_norm'); self.y_norm=mm('y_norm'); self.y_raw=mm('y_raw')
        self.observed=mm('history_observed'); self.valid=mm('future_valid'); self.direct=mm('direct_mask')
        self.original_valid=mm('artificial_missing') if (r/f'{split}_artificial_missing.npy').is_file() else None
        self.tod=mm('tod_hist'); self.dow=mm('dow_hist'); self.meta=pd.read_csv(r/f'{split}_manifest.csv')
        self.n=len(self.meta) if limit is None else min(int(limit),len(self.meta))
        self.window_starts=self.meta.window_start.to_numpy(np.int64)
        self.x_full=None
        if cf_training:
            if split!='train':raise ValueError('Counterfactual full-history supervision is Train-only')
            full_path=r/f'{split}_x_full_norm.npy'
            if not full_path.exists():
                raise FileNotFoundError(f'Counterfactual training input unavailable: {full_path}')
            self.x_full=np.load(full_path,mmap_mode='r')
    def __len__(self): return self.n
    def __getitem__(self,i):
        sample = {
            'index':int(i),'x':torch.from_numpy(np.array(self.x_obs[i],np.float32,copy=True)),
            'mask':torch.from_numpy(np.array(self.observed[i],np.float32,copy=True)),
            'y_norm':torch.from_numpy(np.array(self.y_norm[i],np.float32,copy=True)),
            'y_raw':torch.from_numpy(np.array(self.y_raw[i],np.float32,copy=True)),
            'future_valid':torch.from_numpy(np.array(self.valid[i],np.bool_,copy=True)),
            'direct_mask':torch.from_numpy(np.array(self.direct[i],np.bool_,copy=True)),
            'tod':torch.from_numpy(np.array(self.tod[i],np.int64,copy=True)),
            'dow':torch.from_numpy(np.array(self.dow[i],np.int64,copy=True)),
            'window_start':int(self.window_starts[i]),
            'original_valid': torch.from_numpy((np.asarray(self.observed[i],dtype=bool) | (np.asarray(self.original_valid[i],dtype=bool) if self.original_valid is not None else np.zeros_like(self.observed[i],dtype=bool))).copy()),
        }
        if self.x_full is not None:
            sample['x_full_training_only']=torch.from_numpy(np.array(self.x_full[i],np.float32,copy=True))
        return sample

class RetrievalCache:
    """New-format forecast-conditioned shortlist; incompatible with pre-v2.3 files."""
    def __init__(self, path):
        with np.load(path,allow_pickle=False) as z:
            if int(z['version'])!=25:raise RuntimeError('Use TCRR v2 forecast pool cache')
            self.selected_indices=np.asarray(z['indices'],np.int32)
            self.selected_scores=np.asarray(z['scores'],np.float32)
            self.selected_coverage=np.asarray(z['coverage'],np.float32)
            if not (self.selected_indices.shape==self.selected_scores.shape==self.selected_coverage.shape):
                raise ValueError('v2.3 shortlist arrays disagree')
            self.candidate_count=self.selected_indices.shape[1]
