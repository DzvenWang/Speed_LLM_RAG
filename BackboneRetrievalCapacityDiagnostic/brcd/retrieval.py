from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .masking import broadcast_mask
from .utils import log


@dataclass
class TopKResult:
    indices: np.ndarray  # [Q,K]
    distances: np.ndarray  # [Q,K], squared mean distance
    pool_sizes: np.ndarray  # [Q]


def _temporal_bins(h: int, bins: int) -> list[np.ndarray]:
    if bins < 1 or bins > h:
        raise ValueError(f"temporal_bins must be within [1,H], got bins={bins}, H={h}")
    return [np.asarray(x, dtype=np.int64) for x in np.array_split(np.arange(h), bins)]


def build_embedding(x: np.ndarray, observed: np.ndarray, temporal_bins: int = 3, mask_weight: float = 0.25,
                    chunk_size: int = 2048) -> np.ndarray:
    """Learned-free missing-aware historical state embedding.

    Each temporal bin contributes per-node observed mean plus observed fraction.
    No future value, direct_mask, evaluation label, or backbone output is used.
    """
    q, h, n = x.shape
    bins = _temporal_bins(h, int(temporal_bins))
    out = np.empty((q, len(bins) * n * 2), dtype=np.float32)
    for s in range(0, q, chunk_size):
        e = min(q, s + chunk_size)
        xx = np.asarray(x[s:e], dtype=np.float32)
        mm = np.asarray(observed[s:e], dtype=np.float32)
        pieces = []
        for idx in bins:
            m = mm[:, idx, :]
            v = xx[:, idx, :]
            count = m.sum(axis=1)
            val = (v * m).sum(axis=1) / np.maximum(count, 1.0)
            frac = count / float(len(idx))
            val = np.where(count > 0, val, 0.0)
            pieces.extend([val.astype(np.float32), (float(mask_weight) * frac).astype(np.float32)])
        out[s:e] = np.concatenate(pieces, axis=1)
    return out


def _last_time_features(tod_hist: np.ndarray, dow_hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tod = np.asarray(tod_hist[:, -1] if tod_hist.ndim == 2 else tod_hist, dtype=np.int64)
    dow = np.asarray(dow_hist[:, -1] if dow_hist.ndim == 2 else dow_hist, dtype=np.int64)
    return tod, dow


def _day_type(dow: np.ndarray) -> np.ndarray:
    # 0=weekday, 1=weekend; works for standard 0..6 encoding.
    return (np.asarray(dow, dtype=np.int64) >= 5).astype(np.int8)


def _circular_tod_distance(a: int, b: np.ndarray, slots: int) -> np.ndarray:
    d = np.abs(np.asarray(b, dtype=np.int64) - int(a))
    return np.minimum(d, int(slots) - d)


def _candidate_pool(train_tod, train_dow, q_tod: int, q_dow: int, radius: int, slots: int,
                    match_day_type: bool, min_candidates: int) -> np.ndarray:
    td = _circular_tod_distance(int(q_tod), train_tod, slots) <= int(radius)
    if match_day_type:
        dt = _day_type(train_dow) == _day_type(np.asarray([q_dow]))[0]
        idx = np.flatnonzero(td & dt)
    else:
        idx = np.flatnonzero(td)
    if len(idx) < min_candidates and match_day_type:
        idx = np.flatnonzero(td)
    if len(idx) < min_candidates:
        idx = np.arange(len(train_tod), dtype=np.int64)
    return idx.astype(np.int64, copy=False)


def search_topk(train_emb: np.ndarray, query_emb: np.ndarray,
                train_tod_hist: np.ndarray, train_dow_hist: np.ndarray,
                query_tod_hist: np.ndarray, query_dow_hist: np.ndarray,
                kmax: int = 50, tod_radius: int = 1, tod_slots: int = 288,
                match_day_type: bool = True, batch_queries: int = 64) -> TopKResult:
    """Exact L2 search inside inference-safe temporal candidate pools."""
    train_tod, train_dow = _last_time_features(train_tod_hist, train_dow_hist)
    query_tod, query_dow = _last_time_features(query_tod_hist, query_dow_hist)
    qn = len(query_emb)
    indices = np.empty((qn, kmax), dtype=np.int64)
    distances = np.empty((qn, kmax), dtype=np.float32)
    pool_sizes = np.empty(qn, dtype=np.int32)

    # Cache candidate pools by (tod, day_type/exact dow ignored) to amortise filtering.
    pool_cache: dict[tuple[int, int], np.ndarray] = {}
    query_groups: dict[tuple[int, int], list[int]] = {}
    qdt = _day_type(query_dow)
    for qi in range(qn):
        key = (int(query_tod[qi]), int(qdt[qi]) if match_day_type else -1)
        query_groups.setdefault(key, []).append(qi)

    for gi, (key, qrows_list) in enumerate(query_groups.items(), 1):
        qrows = np.asarray(qrows_list, dtype=np.int64)
        if key not in pool_cache:
            representative = int(qrows[0])
            pool_cache[key] = _candidate_pool(
                train_tod, train_dow, int(query_tod[representative]), int(query_dow[representative]),
                tod_radius, tod_slots, match_day_type, kmax,
            )
        cidx = pool_cache[key]
        c = np.asarray(train_emb[cidx], dtype=np.float32)
        c2 = np.sum(c * c, axis=1, dtype=np.float32)[None, :]
        for s in range(0, len(qrows), int(batch_queries)):
            rows = qrows[s:s + int(batch_queries)]
            q = np.asarray(query_emb[rows], dtype=np.float32)
            q2 = np.sum(q * q, axis=1, dtype=np.float32)[:, None]
            # mean squared L2 keeps distance scale stable across embedding size.
            d = (q2 + c2 - 2.0 * (q @ c.T)) / float(train_emb.shape[1])
            np.maximum(d, 0.0, out=d)
            kk = min(kmax, len(cidx))
            part = np.argpartition(d, kk - 1, axis=1)[:, :kk]
            pd = np.take_along_axis(d, part, axis=1)
            order = np.argsort(pd, axis=1)
            part = np.take_along_axis(part, order, axis=1)
            pd = np.take_along_axis(pd, order, axis=1)
            if kk < kmax:
                pad_n = kmax - kk
                part = np.concatenate([part, np.repeat(part[:, -1:], pad_n, axis=1)], axis=1)
                pd = np.concatenate([pd, np.repeat(pd[:, -1:], pad_n, axis=1)], axis=1)
            indices[rows] = cidx[part[:, :kmax]]
            distances[rows] = pd[:, :kmax].astype(np.float32)
            pool_sizes[rows] = len(cidx)
        if gi % 64 == 0 or gi == len(query_groups):
            log(f"RETRIEVAL_SEARCH groups={gi}/{len(query_groups)} queries_done={sum(len(v) for v in list(query_groups.values())[:gi])}")
    return TopKResult(indices, distances, pool_sizes)


def _weights(dist: np.ndarray, tau: float) -> np.ndarray:
    # Per-query scale makes tau interpretable across severities.
    scale = np.median(dist, axis=1, keepdims=True)
    scale = np.maximum(scale, 1e-6)
    z = -dist / (float(tau) * scale)
    z = z - np.max(z, axis=1, keepdims=True)
    w = np.exp(z, dtype=np.float32)
    return w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)


def aggregate_future(topk: TopKResult, train_y: np.ndarray, train_valid: np.ndarray,
                     k: int, tau: float, chunk_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    q = len(topk.indices)
    k = min(int(k), topk.indices.shape[1])
    p, n = int(train_y.shape[1]), int(train_y.shape[2])
    out = np.full((q, p, n), np.nan, dtype=np.float32)
    out_valid = np.zeros((q, p, n), dtype=np.bool_)
    for s in range(0, q, int(chunk_size)):
        e = min(q, s + int(chunk_size))
        idx = topk.indices[s:e, :k]
        d = topk.distances[s:e, :k]
        w = _weights(d, tau)[:, :, None, None]
        yy = np.asarray(train_y[idx], dtype=np.float32)
        vv = np.asarray(train_valid[idx], dtype=np.bool_)
        finite = np.isfinite(yy)
        mask = vv & finite
        ww = w * mask.astype(np.float32)
        den = ww.sum(axis=1)
        num = (np.where(mask, yy, 0.0) * ww).sum(axis=1)
        good = den > 1e-12
        pred = np.full_like(num, np.nan, dtype=np.float32)
        pred[good] = num[good] / den[good]
        out[s:e] = pred
        out_valid[s:e] = good
    return out, out_valid


def oracle_future(topk: TopKResult, train_y: np.ndarray, train_valid: np.ndarray,
                  query_y: np.ndarray, query_valid: np.ndarray, k: int,
                  extra_mask: np.ndarray | None = None, chunk_size: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ground-truth oracle over retrieved candidates. Analysis only; never deployable."""
    q = len(topk.indices)
    k = min(int(k), topk.indices.shape[1])
    p, n = int(query_y.shape[1]), int(query_y.shape[2])
    out = np.full((q, p, n), np.nan, dtype=np.float32)
    out_valid = np.zeros((q, p, n), dtype=np.bool_)
    chosen = np.full(q, -1, dtype=np.int64)
    for s in range(0, q, int(chunk_size)):
        e = min(q, s + int(chunk_size))
        idx = topk.indices[s:e, :k]
        cy = np.asarray(train_y[idx], dtype=np.float32)
        cv = np.asarray(train_valid[idx], dtype=np.bool_)
        qy = np.asarray(query_y[s:e], dtype=np.float32)[:, None, :, :]
        qv = np.asarray(query_valid[s:e], dtype=np.bool_)[:, None, :, :]
        m = cv & qv & np.isfinite(cy) & np.isfinite(qy)
        if extra_mask is not None:
            em = broadcast_mask(extra_mask[s:e], (e - s, p, n), name="oracle extra_mask")
            m &= em[:, None, :, :]
        abs_err = np.where(m, np.abs(cy - qy), 0.0)
        count = m.sum(axis=(2, 3))
        mae = abs_err.sum(axis=(2, 3)) / np.maximum(count, 1)
        mae[count == 0] = np.inf
        best = np.argmin(mae, axis=1)
        rows = np.arange(e - s)
        selected_idx = idx[rows, best]
        sy = np.asarray(train_y[selected_idx], dtype=np.float32)
        sv = np.asarray(train_valid[selected_idx], dtype=np.bool_) & np.isfinite(sy)
        out[s:e] = np.where(sv, sy, np.nan)
        out_valid[s:e] = sv
        chosen[s:e] = selected_idx
    return out, out_valid, chosen
