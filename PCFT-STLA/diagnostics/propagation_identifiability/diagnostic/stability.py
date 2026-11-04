"""TRAIN-discovery-only directional lag stability, per audited physical directed edge."""
from __future__ import annotations
import numpy as np


def lag_pair_corr(change, valid, edge_index, lag, min_pairs=64, chunk_edges=128):
    """Masked Pearson change(t) -> change(t+lag), no target values beyond input slice."""
    if not (1 <= lag < len(change)):
        raise ValueError('lag out of range')
    src, dst = edge_index
    E = len(src)
    corr = np.full(E, np.nan, dtype=np.float64)
    counts = np.zeros(E, dtype=np.int64)
    for start in range(0, E, chunk_edges):
        sl = slice(start, min(start + chunk_edges, E))
        sx, dy = src[sl], dst[sl]
        a = change[:-lag, sx].astype(np.float64, copy=False)
        b = change[lag:, dy].astype(np.float64, copy=False)
        ok = valid[:-lag, sx] & valid[lag:, dy] & np.isfinite(a) & np.isfinite(b)
        # Mask invalid BEFORE reductions; invalid traffic may contain NaN.
        ax = np.where(ok, a, 0.)
        by = np.where(ok, b, 0.)
        n = ok.sum(0).astype(np.float64)
        sums = (ax.sum(0), by.sum(0), (ax * ax).sum(0),
                (by * by).sum(0), (ax * by).sum(0))
        sx0, sy0, sxx, syy, sxy = sums
        n_safe = np.maximum(n, 1.)
        vx = np.maximum(sxx - sx0 * sx0 / n_safe, 0.)
        vy = np.maximum(syy - sy0 * sy0 / n_safe, 0.)
        den = np.sqrt(vx * vy)
        good = (n >= min_pairs) & (den > 1e-9)
        r = np.full(len(n), np.nan, np.float64)
        r[good] = np.clip((sxy[good] - sx0[good]*sy0[good]/n_safe[good]) / den[good], -1, 1)
        corr[sl] = r
        counts[sl] = n.astype(np.int64)
    return corr, counts


def change_and_valid(traffic, mask):
    x = np.asarray(traffic, np.float32)
    m = np.asarray(mask, bool) & np.isfinite(x)
    d = np.zeros(x.shape, dtype=np.float32)
    dm = np.zeros(m.shape, dtype=bool)
    dm[1:] = m[1:] & m[:-1]
    d[1:] = np.where(dm[1:], x[1:]-x[:-1], 0.)
    return d, dm


def best_from_correlations(corr, counts):
    good = np.isfinite(corr)
    mags = np.where(good, np.abs(corr), -np.inf)
    best_index = np.argmax(mags, axis=0)
    col = np.arange(mags.shape[1])
    has = np.any(good, axis=0)
    best_r = np.where(has, corr[best_index, col], np.nan)
    best_n = np.where(has, counts[best_index, col], 0)
    best_lag = np.where(has, best_index+1, 0)
    return best_r, best_lag, best_n


def compute_stability(traffic, raw_valid, ei, *, blocks=6, max_lag=6,
                      min_pairs=64, abs_corr=0.15, sign_fraction=0.75,
                      lag_fraction=0.67, direction_gap=0.03):
    """Only discovery time range is supplied. No VAL or TEST is read."""
    T, N = traffic.shape
    if T < (max_lag + 2) * blocks * 2:
        raise ValueError('TRAIN discovery segment too short')
    d, dm = change_and_valid(traffic, raw_valid)
    E = ei.shape[1]
    block_stats = []
    cuts = np.linspace(0, T, blocks+1, dtype=int)
    for k in range(blocks+1):
        if k == blocks:
            c, m = d, dm
        else:
            # Boundaries are independent; differences at first block time disabled.
            sl = slice(cuts[k], cuts[k+1])
            c = d[sl].copy(); m = dm[sl].copy()
            m[0] = False; c[0] = 0.
        rr, nn = [], []
        for lag in range(1, max_lag+1):
            r, n = lag_pair_corr(c, m, ei, lag, min_pairs=min_pairs)
            rr.append(r);nn.append(n)
        corr = np.stack(rr);cnt = np.stack(nn)
        br, bl, bn = best_from_correlations(corr,cnt)
        block_stats.append((br, bl, bn, corr))
    full_r, full_lag, full_pairs, full_corr = block_stats[-1]
    block_r = np.stack([x[0] for x in block_stats[:-1]])
    block_lag = np.stack([x[1] for x in block_stats[:-1]])
    avail = np.isfinite(block_r)
    enough = avail.sum(0) >= max(3, int(np.ceil(blocks*.67)))
    signs_match = avail & (np.sign(block_r) == np.sign(full_r)[None])
    sign_agreement = signs_match.sum(0) / np.maximum(avail.sum(0),1)
    lag_match = avail & (np.abs(block_lag - full_lag[None]) <= 1)
    lag_agreement = lag_match.sum(0) / np.maximum(avail.sum(0),1)
    stable = (enough & (np.abs(full_r) >= abs_corr) &
              (sign_agreement >= sign_fraction) &
              (lag_agreement >= lag_fraction))
    e_to_index = {(int(a),int(b)):i for i,(a,b) in enumerate(ei.T)}
    rev_r = np.array([full_r[e_to_index.get((int(b),int(a)),i)] for i,(a,b) in enumerate(ei.T)])
    gap = np.abs(full_r) - np.abs(rev_r)
    directional = stable & np.isfinite(gap) & (gap >= direction_gap)
    return {
        'src': ei[0], 'dst': ei[1], 'best_lag': full_lag, 'corr': full_r,
        'pairs': full_pairs, 'available_blocks':avail.sum(0),
        'sign_fraction': sign_agreement, 'lag_fraction':lag_agreement,
        'reverse_corr':rev_r, 'direction_gap':gap,
        'stable_lag': stable, 'stable_direction_lag':directional,
        'corr_by_lag':full_corr.T.copy(),
        'discovery_timesteps':T,'blocks':blocks,'candidate_edges':E,
    }
