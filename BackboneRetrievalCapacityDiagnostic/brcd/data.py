from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


MASK_EXACT_NAMES = (
    "{split}_history_observed.npy",
    "{split}_history_observed_mask.npy",
    "{split}_x_observed_mask.npy",
    "{split}_x_obs_mask.npy",
    "{split}_obs_mask.npy",
    "{split}_history_mask.npy",
)


def discover_history_mask(cache_dir: str | Path, split: str) -> Path:
    root = Path(cache_dir)
    for template in MASK_EXACT_NAMES:
        p = root / template.format(split=split)
        if p.exists():
            return p
    candidates = []
    for p in root.glob(f"{split}_*.npy"):
        name = p.name.lower()
        if any(bad in name for bad in ("direct", "future", "support", "y_")):
            continue
        if ("observ" in name and "hist" in name) or ("obs" in name and "mask" in name) or "history_mask" in name:
            candidates.append(p)
    if len(candidates) == 1:
        return candidates[0]
    listing = sorted(p.name for p in root.glob(f"{split}_*.npy"))
    raise FileNotFoundError(
        f"Could not uniquely locate inference-time historical observation mask for split={split} in {root}. "
        f"Candidates={candidates}; available={listing}"
    )


@dataclass
class RetrievalSplit:
    cache_dir: Path
    split: str
    x: np.ndarray
    y: np.ndarray
    future_valid: np.ndarray
    direct: np.ndarray
    observed: np.ndarray
    tod_hist: np.ndarray
    dow_hist: np.ndarray
    meta: pd.DataFrame
    mask_path: Path

    @property
    def q(self) -> int:
        return int(self.x.shape[0])

    @property
    def h(self) -> int:
        return int(self.x.shape[1])

    @property
    def n(self) -> int:
        return int(self.x.shape[2])

    @property
    def p(self) -> int:
        return int(self.y.shape[1])

    @property
    def view_ids(self) -> np.ndarray:
        if "view_id" not in self.meta.columns:
            raise RuntimeError(f"{self.cache_dir}/{self.split}_manifest.csv missing view_id")
        return self.meta["view_id"].astype(str).to_numpy()


def _load_bool(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def load_split(cache_dir: str | Path, split: str, max_queries: int | None = None) -> RetrievalSplit:
    root = Path(cache_dir)
    required = {
        "x": root / f"{split}_x_obs_norm.npy",
        "y": root / f"{split}_y_raw.npy",
        "future_valid": root / f"{split}_future_valid.npy",
        "direct": root / f"{split}_direct_mask.npy",
        "tod_hist": root / f"{split}_tod_hist.npy",
        "dow_hist": root / f"{split}_dow_hist.npy",
        "meta": root / f"{split}_manifest.csv",
    }
    for name, p in required.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {name} artifact: {p}")
    mask_path = discover_history_mask(root, split)
    x = np.load(required["x"], mmap_mode="r")
    y = np.load(required["y"], mmap_mode="r")
    fv = _load_bool(required["future_valid"])
    direct = _load_bool(required["direct"])
    obs = _load_bool(mask_path)
    tod = np.load(required["tod_hist"], mmap_mode="r")
    dow = np.load(required["dow_hist"], mmap_mode="r")
    meta = pd.read_csv(required["meta"])
    q = len(x) if max_queries is None else min(len(x), int(max_queries))
    if not (len(y) >= q and len(fv) >= q and len(direct) >= q and len(obs) >= q and len(tod) >= q and len(dow) >= q and len(meta) >= q):
        raise RuntimeError(f"Split artifact row mismatch in {root} split={split}")
    # Slices preserve mmap-backed views where possible.
    x = x[:q]; y = y[:q]; fv = fv[:q]; direct = direct[:q]; obs = obs[:q]; tod = tod[:q]; dow = dow[:q]
    meta = meta.iloc[:q].reset_index(drop=True)
    if x.ndim != 3:
        raise RuntimeError(f"Expected x [Q,H,N], got {x.shape}")
    if obs.shape != x.shape:
        # tolerate a singleton channel
        if obs.ndim == 4 and obs.shape[-1] == 1 and obs.shape[:-1] == x.shape:
            obs = obs[..., 0]
        else:
            raise RuntimeError(f"Historical observation mask shape {obs.shape} != x shape {x.shape}; file={mask_path}")
    if y.ndim == 4 and y.shape[-1] == 1:
        y = y[..., 0]
    if y.ndim != 3:
        raise RuntimeError(f"Expected y [Q,P,N], got {y.shape}")
    return RetrievalSplit(root, split, x, y, fv, direct, obs, tod, dow, meta, mask_path)
