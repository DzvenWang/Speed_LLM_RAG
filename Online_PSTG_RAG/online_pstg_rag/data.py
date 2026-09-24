from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from .contract import cache_dir


class OnlineCache:
    def __init__(self, cfg, stage, split=None):
        self.cfg = cfg; self.stage = stage
        self.split = split or cfg["experiment"]["stream_split"]
        r = cache_dir(cfg, stage)
        names = [
            "x_obs_norm", "y_raw", "y_norm", "future_valid", "direct_mask",
            "support_mask", "tod_hist", "dow_hist",
        ]
        for name in names:
            p = r / f"{self.split}_{name}.npy"
            if not p.exists(): raise FileNotFoundError(p)
            setattr(self, name, np.load(p, mmap_mode="r"))
        mp = r / f"{self.split}_manifest.csv"
        self.meta = pd.read_csv(mp)
        col = cfg["experiment"]["manifest_start_column"]
        if col not in self.meta.columns:
            raise RuntimeError(f"Manifest missing {col}; columns={list(self.meta.columns)}")
        self.starts = np.asarray(self.meta[col], dtype=np.int64)
        n = len(self.x_obs_norm)
        if len(self.meta) != n: raise RuntimeError(f"manifest rows={len(self.meta)} != cache rows={n}")
        for name in names[1:]:
            if len(getattr(self, name)) != n:
                raise RuntimeError(f"{self.split}_{name} rows={len(getattr(self, name))} != {n}")
        self.order = np.argsort(self.starts, kind="stable")
        self.history = int(cfg["experiment"]["history"]); self.horizon = int(cfg["experiment"]["horizon"])

    def __len__(self): return len(self.order)
    def cache_index(self, stream_pos): return int(self.order[int(stream_pos)])
    def start(self, idx): return int(self.starts[int(idx)])
    def origin(self, idx): return self.start(idx) + self.history
    def future_end(self, idx): return self.start(idx) + self.history + self.horizon

    def item(self, idx):
        i = int(idx)
        return {
            "index": i,
            "start": self.start(i),
            "origin": self.origin(i),
            "future_end": self.future_end(i),
            "x_obs": torch.from_numpy(np.asarray(self.x_obs_norm[i], np.float32).copy()),
            "y_raw": torch.from_numpy(np.asarray(self.y_raw[i], np.float32).copy()),
            "y_norm": torch.from_numpy(np.asarray(self.y_norm[i], np.float32).copy()),
            "future_valid": torch.from_numpy(np.asarray(self.future_valid[i], bool).copy()),
            "direct_mask": torch.from_numpy(np.asarray(self.direct_mask[i], bool).copy()),
            "support_mask": torch.from_numpy(np.asarray(self.support_mask[i], bool).copy()),
            "tod_hist": torch.from_numpy(np.asarray(self.tod_hist[i], np.int64).copy()),
            "dow_hist": torch.from_numpy(np.asarray(self.dow_hist[i], np.int64).copy()),
        }
