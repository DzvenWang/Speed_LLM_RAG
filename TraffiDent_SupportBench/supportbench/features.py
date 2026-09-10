from __future__ import annotations

import numpy as np


def _safe_stats(x: np.ndarray, m: np.ndarray, axis=0):
    count = np.maximum(m.sum(axis=axis), 1)
    mean = np.where(m, x, 0.0).sum(axis=axis) / count
    var = np.where(m, (x - np.expand_dims(mean, axis=axis))**2, 0.0).sum(axis=axis) / count
    return mean, np.sqrt(np.maximum(var, 1e-8))


def episode_summary_v1(history_speed: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    """Model-agnostic compact feature for canonical retrieval diagnostics.

    Features intentionally use only current/past observations: per-step global
    mean/std/coverage, per-node mean/last/slope/coverage, and global first-last trend.
    Missing entries are ignored rather than zero-imputed in summary statistics.
    """
    x = np.asarray(history_speed, dtype=np.float32)
    m = np.asarray(obs_mask, dtype=bool) & np.isfinite(x)
    H, N = x.shape
    step_mean = np.zeros(H, np.float32); step_std = np.zeros(H, np.float32); step_cov = m.mean(axis=1).astype(np.float32)
    for t in range(H):
        vals = x[t][m[t]]
        if vals.size:
            step_mean[t] = vals.mean(); step_std[t] = vals.std()
    node_mean = np.zeros(N, np.float32); node_last = np.zeros(N, np.float32); node_slope = np.zeros(N, np.float32); node_cov = m.mean(axis=0).astype(np.float32)
    tt = np.arange(H, dtype=np.float32)
    for n in range(N):
        idx = np.flatnonzero(m[:,n])
        if idx.size:
            vals = x[idx,n]
            node_mean[n] = vals.mean(); node_last[n] = vals[-1]
            if idx.size >= 2:
                a = np.polyfit(tt[idx], vals, 1)[0]
                node_slope[n] = np.float32(a)
    global_trend = np.asarray([step_mean[-1] - step_mean[0]], dtype=np.float32)
    return np.concatenate([step_mean, step_std, step_cov, node_mean, node_last, node_slope, node_cov, global_trend]).astype(np.float32)


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def temporal_summary_v2(history_speed: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    x=np.asarray(history_speed,np.float32); m=np.asarray(obs_mask,bool)&np.isfinite(x)
    H,N=x.shape
    mean=np.zeros(H,np.float32); std=np.zeros(H,np.float32); cov=m.mean(axis=1).astype(np.float32)
    for t in range(H):
        vals=x[t][m[t]]
        if len(vals): mean[t]=float(vals.mean()); std[t]=float(vals.std())
    d1=np.diff(mean,prepend=mean[:1]); trend=np.asarray([mean[-1]-mean[0],float(np.mean(d1)),float(np.std(d1))],np.float32)
    return np.concatenate([mean,std,cov,d1,trend]).astype(np.float32)


def spatial_summary_v2(history_speed: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    x=np.asarray(history_speed,np.float32); m=np.asarray(obs_mask,bool)&np.isfinite(x); H,N=x.shape
    node_mean=np.zeros(N,np.float32); node_last=np.zeros(N,np.float32); node_slope=np.zeros(N,np.float32); node_cov=m.mean(axis=0).astype(np.float32)
    tt=np.arange(H,dtype=np.float32)
    for n in range(N):
        idx=np.flatnonzero(m[:,n])
        if len(idx):
            vals=x[idx,n]; node_mean[n]=float(vals.mean()); node_last[n]=float(vals[-1])
            if len(idx)>=2: node_slope[n]=float(np.polyfit(tt[idx],vals,1)[0])
    return np.concatenate([node_mean,node_last,node_slope,node_cov]).astype(np.float32)


def _weighted_neighbor_series(x: np.ndarray, m: np.ndarray, ids: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    ids=np.asarray(ids,np.int64); valid=ids>=0; ids=ids[valid]
    if not len(ids): return np.zeros(x.shape[0],np.float32),np.zeros(x.shape[0],bool)
    w=np.asarray(weights,np.float32)[valid]; vals=x[:,ids]; vm=m[:,ids]&np.isfinite(vals)
    out=np.zeros(x.shape[0],np.float32); om=np.any(vm,axis=1)
    for t in np.flatnonzero(om):
        wt=np.where(vm[t],w,0.0); den=float(wt.sum())
        if den<=0: wt=np.where(vm[t],1.0,0.0); den=float(wt.sum())
        out[t]=float(np.sum(np.where(vm[t],vals[t],0.0)*wt)/max(den,1e-8))
    return out,om


def propagation_summary_v2(history_speed: np.ndarray, obs_mask: np.ndarray, dependency_ids: np.ndarray, dependency_weights: np.ndarray) -> np.ndarray:
    """Per-target relational summary using the Train-only predictive-neighbour graph."""
    x=np.asarray(history_speed,np.float32); m=np.asarray(obs_mask,bool)&np.isfinite(x); H,N=x.shape
    last_gap=np.zeros(N,np.float32); mean_gap=np.zeros(N,np.float32); slope_gap=np.zeros(N,np.float32); dep_cov=np.zeros(N,np.float32)
    tt=np.arange(H,dtype=np.float32)
    for n in range(N):
        neigh,nm=_weighted_neighbor_series(x,m,dependency_ids[n],dependency_weights[n]); tm=m[:,n]; both=nm&tm
        dep_cov[n]=float(both.mean())
        idx=np.flatnonzero(both)
        if not len(idx): continue
        gap=x[idx,n]-neigh[idx]; mean_gap[n]=float(gap.mean()); last_gap[n]=float(gap[-1])
        if len(idx)>=2: slope_gap[n]=float(np.polyfit(tt[idx],gap,1)[0])
    return np.concatenate([mean_gap,last_gap,slope_gap,dep_cov]).astype(np.float32)


def multi_view_features(history_speed: np.ndarray, obs_mask: np.ndarray, dependency_ids: np.ndarray, dependency_weights: np.ndarray) -> dict[str,np.ndarray]:
    return {
        'temporal':temporal_summary_v2(history_speed,obs_mask),
        'spatial':spatial_summary_v2(history_speed,obs_mask),
        'propagation':propagation_summary_v2(history_speed,obs_mask,dependency_ids,dependency_weights),
    }
