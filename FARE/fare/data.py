from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import CacheSplit


class RestorationDataset(Dataset):
    def __init__(self, cache_dir, split: str, indices=None):
        self.data = CacheSplit(cache_dir, split)
        self.indices = np.arange(len(self.data), dtype=np.int64) if indices is None else np.asarray(indices, np.int64)
    def __len__(self): return len(self.indices)
    def __getitem__(self, k):
        i = int(self.indices[k])
        d = self.data
        return {
            "row": i,
            "x": torch.from_numpy(np.asarray(d.x_obs[i], np.float32).copy()),
            "full": torch.from_numpy(np.asarray(d.x_full[i], np.float32).copy()),
            "observed": torch.from_numpy(np.asarray(d.history_observed[i], np.bool_).copy()),
            "artificial": torch.from_numpy(np.asarray(d.artificial_missing[i], np.bool_).copy()),
            "y_norm": torch.from_numpy(np.asarray(d.y_norm[i], np.float32).copy()),
            "future_valid": torch.from_numpy(np.asarray(d.future_valid[i], np.bool_).copy()),
            "tod": torch.from_numpy(np.asarray(d.tod_hist[i], np.int64).copy()),
            "dow": torch.from_numpy(np.asarray(d.dow_hist[i], np.int64).copy()),
            "window_start": int(d.meta.iloc[i].window_start),
        }
