from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CachedSplit(Dataset):
    def __init__(self, root, split):
        r = Path(root); self.split = split
        self.x = np.load(r/f'{split}_x_obs_norm.npy', mmap_mode='r')
        self.y_raw = np.load(r/f'{split}_y_raw.npy', mmap_mode='r')
        self.y_norm = np.load(r/f'{split}_y_norm.npy', mmap_mode='r')
        self.valid = np.load(r/f'{split}_future_valid.npy', mmap_mode='r')
        self.direct = np.load(r/f'{split}_direct_mask.npy', mmap_mode='r')
        self.support = np.load(r/f'{split}_support_mask.npy', mmap_mode='r')
        self.tod_h = np.load(r/f'{split}_tod_hist.npy', mmap_mode='r')
        self.dow_h = np.load(r/f'{split}_dow_hist.npy', mmap_mode='r')
        self.tod_f = np.load(r/f'{split}_tod_future.npy', mmap_mode='r')
        self.dow_f = np.load(r/f'{split}_dow_future.npy', mmap_mode='r')
        self.meta = pd.read_csv(r/f'{split}_manifest.csv')
    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        return {
            'x': torch.from_numpy(np.asarray(self.x[i]).copy()),
            'y_raw': torch.from_numpy(np.asarray(self.y_raw[i]).copy()),
            'y_norm': torch.from_numpy(np.asarray(self.y_norm[i]).copy()),
            'future_valid': torch.from_numpy(np.asarray(self.valid[i], dtype=np.bool_).copy()),
            'direct_mask': torch.from_numpy(np.asarray(self.direct[i], dtype=np.bool_).copy()),
            'support_mask': torch.from_numpy(np.asarray(self.support[i], dtype=np.bool_).copy()),
            'tod_hist': torch.from_numpy(np.asarray(self.tod_h[i], dtype=np.int64).copy()),
            'dow_hist': torch.from_numpy(np.asarray(self.dow_h[i], dtype=np.int64).copy()),
            'tod_future': torch.from_numpy(np.asarray(self.tod_f[i], dtype=np.int64).copy()),
            'dow_future': torch.from_numpy(np.asarray(self.dow_f[i], dtype=np.int64).copy()),
            'index': int(i),
        }
