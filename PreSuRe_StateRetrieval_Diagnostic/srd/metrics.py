from __future__ import annotations
import numpy as np


def masked_mae(pred, y, valid, node_mask):
    p = np.asarray(pred, np.float32); yy = np.asarray(y, np.float32)
    v = np.asarray(valid, bool); nm = np.asarray(node_mask, bool)
    if p.ndim == 2: p = p[None]
    if yy.ndim == 2: yy = yy[None]
    if v.ndim == 2: v = v[None]
    if nm.ndim == 1: nm = nm[None]
    m = v & nm[:, None, :] & np.isfinite(p) & np.isfinite(yy)
    e = np.abs(p - yy)
    num = np.where(m, e, 0.0).sum(axis=(1, 2), dtype=np.float64); den = m.sum(axis=(1, 2))
    return np.divide(num, den, out=np.full(len(p), np.nan), where=den > 0)


def improvement_pct(base, model):
    if not (np.isfinite(base) and np.isfinite(model)) or base == 0: return float("nan")
    return float((base - model) / base * 100.0)
