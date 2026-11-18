from __future__ import annotations
"""Additive raw-target MAE/RMSE/MAPE for ReShift-STLA v0.2 evaluation.

MAPE protocol: percentage, use original valid forecast cells and |truth| > 1.0
in raw speed/flow units. MAE and RMSE retain their original masks and values.
"""
import json
from pathlib import Path
import numpy as np

MAPE_MIN_TARGET_ABS = 1.0
MAPE_PROTOCOL = 'raw_target_percent_exclude_abs_le_1_v1'


def _scores(pred, true, selection):
    p = np.asarray(pred)
    y = np.asarray(true)
    mask = np.asarray(selection, dtype=bool)
    if p.shape != y.shape or mask.shape != p.shape:
        raise ValueError(f'Metric shape mismatch: pred={p.shape} true={y.shape} mask={mask.shape}')
    m = mask & np.isfinite(p) & np.isfinite(y)
    e = p[m] - y[m]
    n = int(e.size)
    # Preserve exactly the previous MAE/RMSE cell set and arithmetic.
    mae = float(np.mean(np.abs(e))) if n else float('nan')
    rmse = float(np.sqrt(np.mean(e * e))) if n else float('nan')
    raw_y = y[m]
    eligible = np.abs(raw_y) > MAPE_MIN_TARGET_ABS
    mape_n = int(np.count_nonzero(eligible))
    if mape_n:
        # FP64 only in relative error to avoid overflow/round-off at small values.
        rel = np.abs(np.asarray(e[eligible], np.float64)) / np.abs(np.asarray(raw_y[eligible], np.float64))
        mape = float(100.0 * np.mean(rel))
    else:
        mape = float('nan')
    return {'MAE':mae, 'RMSE':rmse, 'MAPE':mape,
            'MAPE_count':mape_n, 'MAPE_coverage': float(mape_n/n) if n else float('nan')}


def metrics(pred, true, valid, direct):
    """Return additive MAPE fields, including masked nodes and step-specific H3/H6/H12.

    Inputs: pred/true/valid [Q,P,N], direct [Q,N].
    """
    p=np.asarray(pred); y=np.asarray(true); v=np.asarray(valid,dtype=bool)
    d=np.asarray(direct,dtype=bool)
    if p.ndim!=3 or y.shape!=p.shape or v.shape!=p.shape or d.shape!=(p.shape[0],p.shape[2]):
        raise ValueError(f'Expected pred/true/valid=[Q,P,N], direct=[Q,N]; got {p.shape}, {y.shape}, {v.shape}, {d.shape}')
    masked=v & d[:,None,:]
    overall=_scores(p,y,v)
    masked_scores=_scores(p,y,masked)
    out={**overall,**{f'directly_masked_{k}':value for k,value in masked_scores.items()}}
    for h in (3,6,12):
        hh=min(h,p.shape[1]); sl=slice(hh-1,hh)
        hs=_scores(p[:,sl],y[:,sl],v[:,sl])
        out.update({f'H{h}_{k}':value for k,value in hs.items()})
        hmask=_scores(p[:,sl],y[:,sl],masked[:,sl])
        out.update({f'H{h}_directly_masked_{k}':value for k,value in hmask.items()})
    return out


def summarize_by_family(pred,true,valid,direct,meta):
    out={'ALL':metrics(pred,true,valid,direct)}
    if 'family' in meta.columns:
        families=meta.family.astype(str).to_numpy()
        for fam in sorted(set(families)):
            ix=np.where(families==fam)[0]
            out[fam]=metrics(pred[ix],true[ix],valid[ix],direct[ix])
    return out


def save_summary(path,payload):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,indent=2,allow_nan=True),encoding='utf-8')
