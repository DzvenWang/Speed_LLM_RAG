from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


class CacheSplit:
    def __init__(self, root: str | Path, split: str):
        r = Path(root); self.root = r; self.split = split
        self.x_obs = np.load(r / f"{split}_x_obs_norm.npy", mmap_mode="r")
        self.x_full = np.load(r / f"{split}_x_full_norm.npy", mmap_mode="r")
        self.history_observed = np.load(r / f"{split}_history_observed.npy", mmap_mode="r")
        self.artificial_missing = np.load(r / f"{split}_artificial_missing.npy", mmap_mode="r")
        self.y_raw = np.load(r / f"{split}_y_raw.npy", mmap_mode="r")
        self.y_norm = np.load(r / f"{split}_y_norm.npy", mmap_mode="r")
        self.future_valid = np.load(r / f"{split}_future_valid.npy", mmap_mode="r")
        self.direct_mask = np.load(r / f"{split}_direct_mask.npy", mmap_mode="r")
        self.tod_hist = np.load(r / f"{split}_tod_hist.npy", mmap_mode="r")
        self.dow_hist = np.load(r / f"{split}_dow_hist.npy", mmap_mode="r")
        self.meta = pd.read_csv(r / f"{split}_manifest.csv")
        manifest = r / "cache_manifest.json"
        self.cache_manifest = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}

    def __len__(self): return len(self.x_obs)

    def original_valid(self, i):
        return np.asarray(self.history_observed[i], bool) | np.asarray(self.artificial_missing[i], bool)
