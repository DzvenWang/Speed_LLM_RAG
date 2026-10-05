from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .masking import broadcast_mask


def _mask(valid, *arrays, extra=None):
    m = np.asarray(valid, dtype=np.bool_).copy()
    for a in arrays:
        m &= np.isfinite(a)
    if extra is not None:
        m &= broadcast_mask(extra, m.shape, name="extra metric mask")
    return m


def mae(pred, y, valid, extra=None) -> float:
    m = _mask(valid, pred, y, extra=extra)
    if not np.any(m):
        return float("nan")
    return float(np.abs(np.asarray(pred)[m] - np.asarray(y)[m]).mean())


def rmse(pred, y, valid, extra=None) -> float:
    m = _mask(valid, pred, y, extra=extra)
    if not np.any(m):
        return float("nan")
    d = np.asarray(pred)[m] - np.asarray(y)[m]
    return float(np.sqrt(np.mean(d * d)))


def summary(pred, y, valid, direct) -> dict:
    return {
        "MAE": mae(pred, y, valid),
        "RMSE": rmse(pred, y, valid),
        "Masked_MAE": mae(pred, y, valid, extra=direct),
        "Masked_RMSE": rmse(pred, y, valid, extra=direct),
    }


def gain_pct(base: float, new: float) -> float:
    if not (math.isfinite(base) and math.isfinite(new)) or base == 0:
        return float("nan")
    return float(100.0 * (base - new) / base)


def per_query_mae(pred, y, valid, extra=None) -> np.ndarray:
    p = np.asarray(pred)
    yy = np.asarray(y)
    m = _mask(valid, p, yy, extra=extra)
    err = np.where(m, np.abs(p - yy), 0.0)
    cnt = m.sum(axis=(1, 2))
    out = err.sum(axis=(1, 2)) / np.maximum(cnt, 1)
    out = out.astype(np.float64)
    out[cnt == 0] = np.nan
    return out


def paired_bootstrap_gain(base_q: np.ndarray, new_q: np.ndarray, n_boot: int = 1000, seed: int = 10) -> dict:
    b = np.asarray(base_q, dtype=np.float64)
    n = np.asarray(new_q, dtype=np.float64)
    ok = np.isfinite(b) & np.isfinite(n)
    b = b[ok]; n = n[ok]
    if len(b) == 0:
        return {"queries": 0, "gain_pct": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    observed = gain_pct(float(b.mean()), float(n.mean()))
    rng = np.random.default_rng(int(seed))
    vals = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        idx = rng.integers(0, len(b), size=len(b))
        vals[i] = gain_pct(float(b[idx].mean()), float(n[idx].mean()))
    return {
        "queries": int(len(b)),
        "gain_pct": float(observed),
        "ci_low": float(np.percentile(vals, 2.5)),
        "ci_high": float(np.percentile(vals, 97.5)),
    }


def rankdata(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    x = np.asarray(list(x), dtype=np.float64)
    y = np.asarray(list(y), dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]; y = y[ok]
    if len(x) < 3:
        return float("nan")
    rx = rankdata(x); ry = rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def fuse(base: np.ndarray, retrieval: np.ndarray, alpha: float) -> np.ndarray:
    b = np.asarray(base, dtype=np.float32)
    r = np.asarray(retrieval, dtype=np.float32)
    out = b.copy()
    good = np.isfinite(r)
    out[good] = (1.0 - float(alpha)) * b[good] + float(alpha) * r[good]
    return out


def oracle_query_fusion_mae(base: np.ndarray, oracle: np.ndarray, y: np.ndarray, valid: np.ndarray,
                            alphas: Iterable[float], extra=None) -> float:
    best = None
    for alpha in alphas:
        pred = fuse(base, oracle, float(alpha))
        q = per_query_mae(pred, y, valid, extra=extra)
        best = q if best is None else np.fmin(best, q)
    if best is None or not np.any(np.isfinite(best)):
        return float("nan")
    return float(np.nanmean(best))
