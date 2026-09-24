from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .contract import cache_dir


class CacheView:
    def __init__(self, cfg, stage, split):
        self.root = cache_dir(cfg, stage); self.split = str(split)
        names = ["x_obs_norm", "x_full_norm", "history_observed", "artificial_missing", "y_raw", "future_valid",
                 "direct_mask", "support_mask", "tod_hist", "dow_hist"]
        for name in names:
            setattr(self, name, np.load(self.root / f"{self.split}_{name}.npy", mmap_mode="r"))
        self.meta = pd.read_csv(self.root / f"{self.split}_manifest.csv")
        n = len(self.x_obs_norm)
        for name in names[1:]:
            if len(getattr(self, name)) != n: raise RuntimeError(f"{split}_{name} length mismatch")

    def __len__(self): return len(self.x_obs_norm)

    def row(self, i):
        i = int(i)
        xo = np.asarray(self.x_obs_norm[i], np.float32)
        xf = np.asarray(self.x_full_norm[i], np.float32)
        obs = np.asarray(self.history_observed[i], bool)
        artificial = np.asarray(self.artificial_missing[i], bool)
        full_valid = obs | (artificial & np.isfinite(xf))
        return {
            "index": i, "x_obs": xo, "x_full": xf, "history_observed": obs,
            "artificial_missing": artificial, "full_valid": full_valid,
            "restorable": artificial & np.isfinite(xf),
            "y_raw": np.asarray(self.y_raw[i], np.float32),
            "future_valid": np.asarray(self.future_valid[i], bool),
            "direct_mask": np.asarray(self.direct_mask[i], bool),
            "support_mask": np.asarray(self.support_mask[i], bool),
            "tod_hist": np.asarray(self.tod_hist[i], np.int64),
            "dow_hist": np.asarray(self.dow_hist[i], np.int64),
        }
