from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .training_modes import require_mode


class CachedSplit(Dataset):
    def __init__(self, root, split, *, input_mode='incomplete'):
        r = Path(root); self.split = split
        self.input_mode = require_mode(input_mode)
        x_file = 'x_full_norm' if self.input_mode == 'complete' else 'x_obs_norm'
        self.x = np.load(r/f'{split}_{x_file}.npy', mmap_mode='r')
        # A complete history removes only *artificial* sensing failures. Genuine
        # source-data invalidity is still excluded, never treated as observed.
        if self.input_mode == 'complete':
            self.history_observed = np.load(r/f'{split}_history_observed.npy', mmap_mode='r')
            self.artificial_missing = np.load(r/f'{split}_artificial_missing.npy', mmap_mode='r')
            if self.x.shape != self.history_observed.shape or self.x.shape != self.artificial_missing.shape:
                raise ValueError(f'Invalid full-history/mask shape: {r} split={split}')
        self.y_raw = np.load(r/f'{split}_y_raw.npy', mmap_mode='r')
        self.y_norm = np.load(r/f'{split}_y_norm.npy', mmap_mode='r')
        self.valid = np.load(r/f'{split}_future_valid.npy', mmap_mode='r')
        self.direct = np.load(r/f'{split}_direct_mask.npy', mmap_mode='r')
        self.tod_h = np.load(r/f'{split}_tod_hist.npy', mmap_mode='r')
        self.dow_h = np.load(r/f'{split}_dow_hist.npy', mmap_mode='r')
        self.tod_f = np.load(r/f'{split}_tod_future.npy', mmap_mode='r')
        self.dow_f = np.load(r/f'{split}_dow_future.npy', mmap_mode='r')
        self.meta = pd.read_csv(r/f'{split}_manifest.csv')
    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        x = np.asarray(self.x[i], dtype=np.float32)
        if self.input_mode == 'complete':
            # materializer: observed = raw_valid & ~artificial_missing;
            # artificial_missing is generated only at originally-valid positions.
            raw_valid_history = (
                np.asarray(self.history_observed[i], dtype=bool)
                | np.asarray(self.artificial_missing[i], dtype=bool)
            )
            if not np.isfinite(x[raw_valid_history]).all():
                raise ValueError(f'Nonfinite valid full-history input split={self.split} row={i}')
            x = np.where(raw_valid_history, x, 0.0).astype(np.float32, copy=False)
        return {
            'x': torch.from_numpy(np.array(x, dtype=np.float32, copy=True)),
            'y_raw': torch.from_numpy(np.asarray(self.y_raw[i]).copy()),
            'y_norm': torch.from_numpy(np.asarray(self.y_norm[i]).copy()),
            'future_valid': torch.from_numpy(np.asarray(self.valid[i], dtype=np.bool_).copy()),
            'direct_mask': torch.from_numpy(np.asarray(self.direct[i], dtype=np.bool_).copy()),
            'tod_hist': torch.from_numpy(np.asarray(self.tod_h[i], dtype=np.int64).copy()),
            'dow_hist': torch.from_numpy(np.asarray(self.dow_h[i], dtype=np.int64).copy()),
            'tod_future': torch.from_numpy(np.asarray(self.tod_f[i], dtype=np.int64).copy()),
            'dow_future': torch.from_numpy(np.asarray(self.dow_f[i], dtype=np.int64).copy()),
            'index': int(i),
        }
