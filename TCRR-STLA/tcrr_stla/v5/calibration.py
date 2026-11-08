"""Chronological model selection, threshold calibration and locked checking.

No distribution-free guarantee is asserted. All thresholds use CALIBRATION only;
CHECK is reported exactly once after model/stage choice and is never used to
select or activate a rejection threshold.
"""
from __future__ import annotations
import hashlib
import numpy as np
import torch


def chronological_partitions(window_starts, history, horizon, selection_fraction=.60,
                             calibration_fraction=.20, smoke=False):
    starts=np.asarray(window_starts,dtype=np.int64).reshape(-1)
    if smoke:
        indices=list(range(len(starts)))
        return dict(selection=indices,calibration=indices,check=indices,
                    smoke_overlap=True,purge_timesteps=0)
    unique=np.unique(starts)
    if unique.size<12:
        raise ValueError('Too few distinct validation windows for disjoint chronological calibration')
    n1=max(1,int(unique.size*selection_fraction))
    n2=max(n1+1,int(unique.size*(selection_fraction+calibration_fraction)))
    if n2>=unique.size:raise ValueError('Empty chronological check split')
    # Exclude neighboring windows across boundaries to avoid shared input/
    # prediction time ranges (the cache window_start is in original time steps).
    gap=int(history)+int(horizon)
    edges=(int(unique[n1]),int(unique[n2]))
    selection=np.flatnonzero(starts < edges[0]-gap).tolist()
    calibration=np.flatnonzero((starts >= edges[0]) & (starts < edges[1]-gap)).tolist()
    check=np.flatnonzero(starts >= edges[1]).tolist()
    if min(len(selection),len(calibration),len(check))<1:
        raise ValueError('Chronological split empty after time-window purging')
    # Stronger than merely disjoint row IDs: no temporal overlap of windows.
    assert max(starts[selection])+gap < min(starts[calibration])
    assert max(starts[calibration])+gap < min(starts[check])
    return dict(selection=selection,calibration=calibration,check=check,
                smoke_overlap=False,purge_timesteps=gap,
                boundaries=dict(calibration_start=edges[0],check_start=edges[1]),
                count_unique_window_starts=int(unique.size),
                window_starts_sha256=hashlib.sha256(starts.astype('<i8').tobytes()).hexdigest())


def choose_rejection_thresholds(base,proposal,gate,predicted_net,harm_probability,y,valid,
                                calibration_fraction=None,min_coverage=.55):
    """Fit on CALIBRATION ONLY. Inputs are raw-unit forecasts and outcomes."""
    base=base.flatten().float();proposal=proposal.flatten().float()
    gate=gate.flatten().float();net=predicted_net.flatten().float()
    risk=harm_probability.flatten().float();y=y.flatten().float()
    v=valid.flatten().bool()
    base,proposal,gate,net,risk,y=[x[v] for x in (base,proposal,gate,net,risk,y)]
    size=base.numel()
    if size<100:
        return dict(min_utility=float('-inf'),max_harm_probability=1.,enabled=False,
                    reason='insufficient_calibration_positions',calibration_coverage=1.,
                    calibration_positions=int(size))
    full=(base+gate*proposal-y).abs()
    no_correction=(base-y).abs()
    tchoices=[float('-inf')]+[float(torch.quantile(net,q)) for q in (.05,.20,.40,.60,.80)]
    rchoices=[1.]+[float(torch.quantile(risk,q)) for q in (.95,.80,.60,.40)]
    baseline=float(full.mean());best_score=baseline
    best=(float('-inf'),1.,1.)
    for threshold in tchoices:
        for rmax in rchoices:
            accept=(net>=threshold)&(risk<=rmax)
            coverage=float(accept.float().mean())
            if coverage<min_coverage:continue
            score=float(torch.where(accept,full,no_correction).mean())
            if score<best_score-1e-8:
                best_score=score;best=(threshold,rmax,coverage)
    threshold,rmax,coverage=best
    enabled=(threshold!=float('-inf') or rmax<1.)
    return dict(min_utility=threshold if enabled else float('-inf'),
                max_harm_probability=rmax if enabled else 1.,enabled=enabled,
                calibration_coverage=coverage if enabled else 1.,
                calibration_positions=int(size),calibration_mae_before=baseline,
                calibration_mae_after=best_score,
                empirical_only=True,selection_used=False,check_used=False)


def assess_rejection_thresholds(base,proposal,gate,predicted_net,harm_probability,y,valid,thresholds):
    """REPORT ONLY on disjoint CHECK; never modifies threshold decisions."""
    v=valid.flatten().bool()
    base,proposal,gate,net,risk,y=[a.flatten().float()[v] for a in
        (base,proposal,gate,predicted_net,harm_probability,y)]
    accept=((net>=thresholds['min_utility']) &
            (risk<=thresholds['max_harm_probability'])) if thresholds.get('enabled') else torch.ones_like(net,dtype=torch.bool)
    before=(base+gate*proposal-y).abs()
    after=torch.where(accept,before,(base-y).abs())
    return dict(check_positions=int(v.sum()),check_coverage=float(accept.float().mean()),
                check_mae_before=float(before.mean()),check_mae_after=float(after.mean()))
