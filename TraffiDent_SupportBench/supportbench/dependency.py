from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .utils import ensure_dir, log


def _corr(a: np.ndarray, b: np.ndarray, m: np.ndarray, min_pairs: int) -> float | None:
    idx = np.flatnonzero(m & np.isfinite(a) & np.isfinite(b))
    if len(idx) < int(min_pairs):
        return None
    aa = np.asarray(a[idx], np.float64)
    bb = np.asarray(b[idx], np.float64)
    sa = float(np.std(aa))
    sb = float(np.std(bb))
    if sa < 1e-8 or sb < 1e-8:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def _best_lag_corr_sampled_raw(
    source: np.ndarray,
    target: np.ndarray,
    source_valid: np.ndarray,
    target_valid: np.ndarray,
    *,
    max_lag_steps: int,
    sample_stride_steps: int,
    min_pairs: int,
) -> tuple[float, int, int]:
    """Best source-leading-target correlation at *raw* time-step lags.

    The sampling cadence and the lag resolution are deliberately separated.
    ``sample_stride_steps`` only chooses sparse anchor times to reduce compute;
    ``lag`` is always applied on the original raw series.  With 5-min data,
    lag=1/2/3 therefore means 5/10/15 minutes even when anchors are sampled
    every 12 raw steps (60 minutes).
    """
    source = np.asarray(source)
    target = np.asarray(target)
    sv = np.asarray(source_valid, dtype=bool)
    tv = np.asarray(target_valid, dtype=bool)
    if not (len(source) == len(target) == len(sv) == len(tv)):
        raise ValueError('source/target/valid series must have the same length')

    max_lag = max(0, int(max_lag_steps))
    stride = max(1, int(sample_stride_steps))
    # Common anchors ensure every tested lag uses the same nominal anchor set.
    stop = len(source) - max_lag
    if stop <= 0:
        return 0.0, 0, 0
    anchors = np.arange(0, stop, stride, dtype=np.int64)

    best_corr = 0.0
    best_lag = 0
    best_n = 0
    found = False
    for lag in range(max_lag + 1):
        src_idx = anchors
        tgt_idx = anchors + lag
        a = source[src_idx]
        b = target[tgt_idx]
        m = sv[src_idx] & tv[tgt_idx]
        c = _corr(a, b, m, min_pairs)
        if c is None:
            continue
        n = int(np.count_nonzero(m & np.isfinite(a) & np.isfinite(b)))
        if (not found) or abs(float(c)) > abs(float(best_corr)):
            found = True
            best_corr = float(c)
            best_lag = int(lag)
            best_n = n
    return best_corr, best_lag, best_n


def build_train_predictive_dependency(
    traffic: np.ndarray,
    valid: np.ndarray,
    speed_idx: int,
    train_start: int,
    train_end: int,
    upstream_ids: np.ndarray,
    upstream_weights: np.ndarray,
    downstream_ids: np.ndarray,
    downstream_weights: np.ndarray,
    out_path: str | Path,
    *,
    max_neighbors: int = 8,
    max_lag_steps: int = 3,
    sample_stride: int = 12,
    min_pairs: int = 64,
    corr_floor: float = 0.05,
    graph_weight_mix: float = 0.20,
    freq_minutes: int = 5,
) -> dict[str, Any]:
    """Build a Train-only predictive-neighbour graph.

    Candidates are restricted to physical directed neighbours.  Predictive
    strength is estimated by source-leading-target correlations at raw lags
    ``0..max_lag_steps`` and mixed with static graph affinity as a stabiliser.

    Important v1.2.1 semantics
    --------------------------
    ``sample_stride`` is *only* a compute-thinning cadence for correlation
    anchors.  It does not redefine lag duration.  For 5-min traffic data and
    ``max_lag_steps=3``, tested lags are 0/5/10/15 minutes even if
    ``sample_stride=12`` (anchors every 60 minutes).

    Validation/Test data and future labels are never accessed.
    """
    s = int(train_start)
    e = int(train_end)
    stride = max(1, int(sample_stride))
    lag_minutes = max(1, int(freq_minutes))

    # Materialise only the Train speed channel.  This keeps raw 5-min temporal
    # resolution while avoiding repeated multi-channel memmap indexing.
    x = np.asarray(traffic[s:e, :, int(speed_idx)], np.float32)
    m = np.asarray(valid[s:e, :, int(speed_idx)], bool)
    n = x.shape[1]
    k = max(1, int(max_neighbors))

    ids = np.full((n, k), -1, np.int64)
    weights = np.zeros((n, k), np.float32)
    lags = np.full((n, k), -1, np.int16)
    corrs = np.zeros((n, k), np.float32)
    relation = np.zeros((n, k), np.int8)
    centrality = np.zeros(n, np.float64)
    strength = np.zeros(n, np.float64)

    for target in range(n):
        cand = []
        for rel, arr, warr in [
            (-1, upstream_ids, upstream_weights),
            (1, downstream_ids, downstream_weights),
        ]:
            for pos, node in enumerate(arr[target]):
                node = int(node)
                if node < 0:
                    continue
                gw = float(warr[target, pos]) if pos < warr.shape[1] else 0.0
                c, lag, npairs = _best_lag_corr_sampled_raw(
                    x[:, node],
                    x[:, target],
                    m[:, node],
                    m[:, target],
                    max_lag_steps=max_lag_steps,
                    sample_stride_steps=stride,
                    min_pairs=min_pairs,
                )
                corr_strength = max(abs(float(c)), float(corr_floor))
                raw = (
                    (1.0 - float(graph_weight_mix)) * corr_strength
                    + float(graph_weight_mix) * max(gw, 1e-6)
                )
                cand.append((raw, node, lag, c, rel, npairs))

        cand.sort(key=lambda z: (-z[0], z[1]))
        cand = cand[:k]
        if not cand:
            continue
        raw = np.asarray([z[0] for z in cand], np.float64)
        raw = raw / max(float(raw.sum()), 1e-12)
        for j, (rw, node, lag, c, rel, npairs) in enumerate(cand):
            ids[target, j] = int(node)
            weights[target, j] = float(raw[j])
            lags[target, j] = int(lag)
            corrs[target, j] = float(c)
            relation[target, j] = int(rel)
            centrality[int(node)] += float(raw[j])
        strength[target] = float(np.sum([z[0] for z in cand]))

    if centrality.max() > 0:
        centrality = centrality / centrality.max()
    if strength.max() > 0:
        strength = strength / strength.max()

    lag_minutes_arr = np.where(lags >= 0, lags.astype(np.int32) * lag_minutes, -1).astype(np.int16)
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    np.savez_compressed(
        out_path,
        ids=ids,
        weights=weights,
        lags=lags,
        lag_minutes=lag_minutes_arr,
        corr=corrs,
        relation=relation,
        source_centrality=centrality.astype(np.float32),
        target_strength=strength.astype(np.float32),
        sample_stride=np.int32(stride),
        sample_stride_steps=np.int32(stride),
        sample_stride_minutes=np.int32(stride * lag_minutes),
        lag_step_minutes=np.int32(lag_minutes),
        train_start=np.int64(s),
        train_end=np.int64(e),
    )
    meta = {
        'nodes': int(n),
        'max_neighbors': int(k),
        'sample_stride': int(stride),
        'sample_stride_steps': int(stride),
        'sample_stride_minutes': int(stride * lag_minutes),
        'lag_resolution_steps': 1,
        'lag_step_minutes': int(lag_minutes),
        'tested_lag_steps': list(range(int(max_lag_steps) + 1)),
        'tested_lag_minutes': [int(i * lag_minutes) for i in range(int(max_lag_steps) + 1)],
        'max_lag_steps': int(max_lag_steps),
        'max_lag_minutes': int(max_lag_steps) * lag_minutes,
        'min_pairs': int(min_pairs),
        'nonempty_targets': int(np.sum(np.any(ids >= 0, axis=1))),
        'mean_source_centrality': float(centrality.mean()),
        'max_source_centrality': float(centrality.max()),
        'semantics': 'sample_stride thins correlation anchors; lag is measured on raw traffic steps',
    }
    log(
        'predictive dependency built: '
        f"targets={meta['nonempty_targets']}/{n}; "
        f"anchors_every={stride} steps ({meta['sample_stride_minutes']} min); "
        f"lags={meta['tested_lag_minutes']} min"
    )
    return meta


def dependency_context_loss(mask: np.ndarray, dep_ids: np.ndarray, dep_weights: np.ndarray) -> np.ndarray:
    """Per-target fraction of predictive-neighbour history removed by a sensing mask."""
    a = np.asarray(mask, bool)
    ids = np.asarray(dep_ids, np.int64)
    w = np.asarray(dep_weights, np.float32)
    n = ids.shape[0]
    out = np.zeros(n, np.float32)
    miss = a.mean(axis=0)
    for t in range(n):
        good = ids[t] >= 0
        if not np.any(good):
            continue
        ii = ids[t, good]
        ww = w[t, good].astype(np.float64)
        den = float(ww.sum())
        if den <= 0:
            ww = np.ones(len(ii), np.float64)
            den = float(len(ii))
        out[t] = float(np.sum(ww * miss[ii]) / max(den, 1e-12))
    return out


def summarize_dependency_context_loss(
    per_target_loss: np.ndarray,
    *,
    quantile: float = 0.75,
) -> dict[str, float]:
    """Summarise per-target predictive-context loss with explicit semantics.

    ``network_mean_context_loss`` is the formal Chapter-3 quantity

        C_q^loss = (1/N) * sum_n C_{q,n}^loss.

    ``context_loss_quantile_value`` retains the historical v1.2.2 diagnostic
    (the configured quantile across targets).  The two quantities are not
    interchangeable and can move in different directions when a mask spreads
    moderate loss across many targets instead of concentrating severe loss on
    a small subset.
    """
    loss = np.asarray(per_target_loss, np.float64).reshape(-1)
    if not len(loss):
        return {
            'network_mean_context_loss': 0.0,
            'context_loss_quantile': float(quantile),
            'context_loss_quantile_value': 0.0,
        }
    q = float(np.clip(float(quantile), 0.0, 1.0))
    return {
        'network_mean_context_loss': float(np.mean(loss)),
        'context_loss_quantile': q,
        'context_loss_quantile_value': float(np.quantile(loss, q)),
    }
