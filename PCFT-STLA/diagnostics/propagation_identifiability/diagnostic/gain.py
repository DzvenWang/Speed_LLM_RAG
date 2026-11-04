"""STLA forecast residual added-value check using s40 observed-only TRAIN views.

Source change time is aligned with the discovery lag L:
  target horizon h uses observed source difference at t_origin+h-L.
  Thus only horizons 1..L are evaluated; h>L requires unknown future source state.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import DataLoader, Subset


# Deliberately explicit: preflight and training must share the same feature contract.
LOCAL_FEATURE_NAMES = (
    'intercept',
    'target_recent_change', 'target_recent_pair_valid',
    'target_lagged_change', 'target_lagged_pair_valid',
    'target_last_observed_value', 'target_last_observed_valid',
)
LOCAL_FEATURE_DIM = len(LOCAL_FEATURE_NAMES)  # seven (7), not six


@dataclass
class ForecastArrays:
    starts: np.ndarray      # [R]
    pred: np.ndarray        # [R,P,N] RAW units
    truth: np.ndarray       # [R,P,N] RAW units
    future_valid: np.ndarray  # [R,P,N]
    x: np.ndarray           # [R,H,N] normalized obs
    observed: np.ndarray    # [R,H,N]


def choose_unique_windows(ds, max_rows=10000):
    # Fixed deterministic choice, one sensing view per window; never merge masks.
    starts = ds.meta['window_start'].to_numpy(np.int64)
    order = np.argsort(starts, kind='stable')
    _, first = np.unique(starts[order], return_index=True)
    ids = order[first]
    if len(ids) > max_rows and max_rows > 0:
        ids = ids[np.unique(np.linspace(0, len(ids)-1, max_rows, dtype=np.int64))]
    return np.asarray(ids, np.int64)


@torch.inference_mode()
def predict_cached(adapter, ds, row_ids, batch_size=16):
    loader = DataLoader(Subset(ds, row_ids.tolist()), batch_size=batch_size,
                        shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    adapter.model.eval()
    predictions = []
    for step, batch in enumerate(loader):
        pr = adapter.raw(adapter.eval_forward(batch)).detach().float().cpu().numpy()
        if not np.isfinite(pr).all() or pr.shape[1:] != tuple(ds.y_raw.shape[1:]):
            raise RuntimeError(f'STLA nonfinite/shape mismatch at step {step}: {pr.shape}')
        predictions.append(pr.astype(np.float32, copy=False))
        if (step+1) % 100 == 0:
            now=datetime.now().astimezone().isoformat(timespec='seconds')
            print(f'[{now}] STLA_INFERENCE progress_batches={step+1}/{len(loader)}', flush=True)
    if not predictions: raise RuntimeError('No STLA inference batches; empty split or invalid sample budget')
    p = np.concatenate(predictions)
    x = np.asarray(ds.x[row_ids], np.float32).copy()
    mask_path = ds.root / f'{ds.split}_history_observed.npy' if hasattr(ds, 'root') else None
    if mask_path is None:
        raise RuntimeError('CachedSplit.root missing; provide dataset cache directory explicitly')
    observed = np.asarray(np.load(mask_path, mmap_mode='r')[row_ids], bool).copy()
    return ForecastArrays(ds.meta['window_start'].to_numpy(np.int64)[row_ids], p,
                          np.asarray(ds.y_raw[row_ids], np.float32).copy(),
                          np.asarray(ds.valid[row_ids], bool).copy(),
                          x, observed)


def ridge_fit(X, y, alpha=1.0):
    # Unpenalized intercept. Stronger shrinkage is desired for tiny residual gains.
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    gram = X.T @ X
    penalty = np.eye(X.shape[1], dtype=np.float64) * alpha
    penalty[0,0] = 0.
    return np.linalg.solve(gram + penalty + np.eye(X.shape[1]) * 1e-9, X.T @ y)


def edge_horizon_features(arr: ForecastArrays, src, dst, lag, h):
    if not (1 <= h <= lag <= arr.x.shape[1]-1):
        raise ValueError('Invalid horizon / lag; source sample would be unobserved future')
    k = lag-h+1
    xs = arr.x[:, -k, src]; xs_prev = arr.x[:, -k-1, src]
    os = arr.observed[:, -k, src] & arr.observed[:, -k-1, src]
    dst_recent = arr.x[:, -1, dst] - arr.x[:, -2, dst]
    m_recent = arr.observed[:, -1, dst] & arr.observed[:, -2, dst]
    dst_lag = arr.x[:, -k, dst] - arr.x[:, -k-1, dst]
    m_lag = arr.observed[:, -k, dst] & arr.observed[:, -k-1, dst]
    last_target = arr.x[:, -1, dst]
    last_valid = arr.observed[:, -1, dst]
    local = np.column_stack([
        np.ones(len(xs), np.float32),
        np.where(m_recent, dst_recent, 0.), m_recent.astype(np.float32),
        np.where(m_lag, dst_lag, 0.), m_lag.astype(np.float32),
        np.where(last_valid, last_target, 0.),last_valid.astype(np.float32)
    ]).astype(np.float64)
    add = np.where(os, xs-xs_prev, 0.).astype(np.float64)
    valid = os & arr.future_valid[:, h-1, dst] & np.isfinite(arr.truth[:,h-1,dst])
    return local, add, valid


def audit_forecast_arrays(arr: ForecastArrays, *, split: str, max_lag: int = 6) -> dict:
    """Check the exact feature, observed-history, prediction and target contracts.

    Preflight calls the same edge_horizon_features() that full gain analysis uses;
    no second, stale hard-coded dimension is allowed. Does not inspect future
    source values or TEST. Works with one genuine TRAIN/VAL sample.
    """
    R = len(arr.starts)
    if R < 1 or arr.pred.ndim != 3:
        raise RuntimeError(f'{split}: missing forecast rows or malformed prediction shape')
    _, P, N = arr.pred.shape
    H = arr.x.shape[1] if arr.x.ndim == 3 else -1
    checks = {
        'prediction': (arr.pred, (R, P, N)),
        'truth': (arr.truth, (R, P, N)),
        'future_valid': (arr.future_valid, (R, P, N)),
        'x': (arr.x, (R, H, N)),
        'observed': (arr.observed, (R, H, N)),
    }
    if N < 2 or H < 2 or P < 1 or not 1 <= max_lag < H:
        raise RuntimeError(f'{split}: incompatible H/P/N or max_lag: {(H,P,N,max_lag)}')
    for name, (array, shape) in checks.items():
        if array.shape != shape:
            raise RuntimeError(f'{split}: {name} shape {array.shape} expected {shape}')
    if not np.isfinite(arr.pred).all():
        raise RuntimeError(f'{split}: nonfinite model prediction')
    # A small but meaningful test matrix. In addition to lag=1, check a later
    # horizon to catch time offsets and an out-of-range future-source path.
    checked = 0
    for lag in sorted(set((1, min(3, max_lag), max_lag))):
        for horizon in sorted(set((1, lag))):
            local, source, eligible = edge_horizon_features(arr, 0, 1, lag, horizon)
            if local.shape != (R, LOCAL_FEATURE_DIM):
                raise RuntimeError(f'{split}: local features at (lag={lag},h={horizon}) have '
                                   f'shape={local.shape}, expected={(R, LOCAL_FEATURE_DIM)}')
            if source.shape != (R,) or eligible.shape != (R,):
                raise RuntimeError(f'{split}: source/eligibility shape mismatch at lag={lag},h={horizon}')
            if not np.isfinite(local).all() or not np.isfinite(source).all():
                raise RuntimeError(f'{split}: nonfinite feature at lag={lag},h={horizon}')
            k = lag - horizon + 1
            valid_pair = arr.observed[:, -k, 0] & arr.observed[:, -k-1, 0]
            expected_eligible = (valid_pair & arr.future_valid[:,horizon-1,1] &
                                 np.isfinite(arr.truth[:,horizon-1,1]))
            if not np.array_equal(eligible, expected_eligible):
                raise RuntimeError(f'{split}: source observation/target eligibility misalignment')
            checked += 1
    return {'split':split,'rows':R,'nodes':N,'history_steps':H,'horizon_steps':P,
            'local_feature_dim':LOCAL_FEATURE_DIM,'checked_pairs':checked}


def gain_for_candidate(fit:ForecastArrays, evaluate:ForecastArrays, src,dst,lag,
                       *, alpha=3.0,min_fit=60,min_eval=30, group_count=4):
    """Strictly fit on early TRAIN only; evaluate is never used for coefficients."""
    overall = dict(raw_abs=0.,local_abs=0.,plus_abs=0.,count=0,
                   horizons_evaluated=0,group_local=np.zeros(group_count),
                   group_plus=np.zeros(group_count),group_n=np.zeros(group_count,int))
    per_h=[]
    for h in range(1,lag+1):
        Xf, sf, mf = edge_horizon_features(fit,src,dst,lag,h)
        Xe, se, me = edge_horizon_features(evaluate,src,dst,lag,h)
        if int(mf.sum())<min_fit or int(me.sum())<min_eval:
            continue
        yfit=(fit.truth[:,h-1,dst] - fit.pred[:,h-1,dst])[mf]
        y_eval=(evaluate.truth[:,h-1,dst] - evaluate.pred[:,h-1,dst])[me]
        Xfit=Xf[mf]; Xeval=Xe[me]; fs=sf[mf]; es=se[me]
        beta_local=ridge_fit(Xfit,yfit,alpha)
        beta_plus=ridge_fit(np.column_stack([Xfit,fs]),yfit,alpha)
        d_raw=np.abs(y_eval)
        d_local=np.abs(y_eval-Xeval@beta_local)
        d_plus=np.abs(y_eval-np.column_stack([Xeval,es])@beta_plus)
        overall['raw_abs']+=float(d_raw.sum())
        overall['local_abs']+=float(d_local.sum())
        overall['plus_abs']+=float(d_plus.sum())
        overall['count']+=int(len(d_raw));overall['horizons_evaluated']+=1
        # Group by actual evaluation time (not arbitrary batch order).
        idx=np.flatnonzero(me)
        lo,hi=int(evaluate.starts.min()),int(evaluate.starts.max())+1
        gid=np.minimum(group_count-1, (group_count*(evaluate.starts[idx]-lo)//max(1,hi-lo)).astype(int))
        np.add.at(overall['group_local'],gid,d_local)
        np.add.at(overall['group_plus'],gid,d_plus)
        np.add.at(overall['group_n'],gid,1)
        per_h.append((h,len(d_raw),float(d_raw.mean()),float(d_local.mean()),float(d_plus.mean())))
    n=overall['count']
    if not n:
        return dict(eligible=False,n=0,horizons=0,mae_raw=None,mae_local=None,
                    mae_plus=None,gain_vs_local_pct=None,gain_vs_stla_pct=None,
                    positive_groups=0,groups_with_data=0,per_horizon=[])
    raw=overall['raw_abs']/n;local=overall['local_abs']/n;plus=overall['plus_abs']/n
    group_ok=(overall['group_n']>=min_eval)
    group_gains=overall['group_local']-overall['group_plus']
    return dict(eligible=True,n=n,horizons=overall['horizons_evaluated'],
                mae_raw=raw,mae_local=local,mae_plus=plus,
                gain_vs_local_pct=100.*(local-plus)/max(local,1e-9),
                gain_vs_stla_pct=100.*(raw-plus)/max(raw,1e-9),
                positive_groups=int(((group_gains>0)&group_ok).sum()),
                groups_with_data=int(group_ok.sum()),
                per_horizon=[{'h':h,'n':nr,'mae_raw':br,'mae_local':bl,'mae_plus':bp} for h,nr,br,bl,bp in per_h])
