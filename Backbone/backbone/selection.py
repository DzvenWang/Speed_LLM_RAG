from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

MODELS = ['STAEformer', 'HimNet', 'BigST', 'STDN', 'PatchSTG']


def latest_summary(exp_root, model, subgraph, stage):
    root = Path(exp_root) / model / subgraph / stage
    files = sorted(root.glob('*/summary.json')) if root.exists() else []
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def tie_aware_support_rank(df: pd.DataFrame, tie_rel: float) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()
    d = df.sort_values('val_support_MAE').copy().reset_index(drop=True)
    groups = []
    group = 0
    anchor = None
    for val in d.val_support_MAE.astype(float):
        if anchor is None:
            anchor = val
        elif val > anchor * (1.0 + tie_rel):
            group += 1
            anchor = val
        groups.append(group)
    d['support_tie_group'] = groups
    return d.sort_values(
        ['support_tie_group', 'val_masked_MAE', 'val_inference_ms_per_query', 'parameters', 'val_MAE'],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)


def screening_shortlist(df: pd.DataFrame, overall_tol: float, tie_rel: float, k: int):
    best = float(df.val_MAE.min())
    d = df.copy()
    d['within_overall_tolerance'] = d.val_MAE <= best * (1.0 + overall_tol)
    eligible = tie_aware_support_rank(d[d.within_overall_tolerance].copy(), tie_rel)
    selected = list(eligible.model.astype(str))[:k]
    if len(selected) < k:
        rest = d[~d.model.astype(str).isin(selected)].sort_values(
            ['val_MAE', 'val_support_MAE', 'val_inference_ms_per_query', 'parameters']
        )
        selected.extend(list(rest.model.astype(str))[:k-len(selected)])
    d['shortlisted'] = d.model.astype(str).isin(selected)
    return d, selected, best


def formal_aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    agg = rows.groupby('model', as_index=False).agg(
        n_seeds=('seed', 'count'),
        mean_val_MAE=('val_MAE', 'mean'),
        std_val_MAE=('val_MAE', 'std'),
        mean_val_support_MAE=('val_support_MAE', 'mean'),
        std_val_support_MAE=('val_support_MAE', 'std'),
        mean_val_masked_MAE=('val_masked_MAE', 'mean'),
        mean_val_H12_MAE=('val_H12_MAE', 'mean'),
        mean_inference_ms_per_query=('val_inference_ms_per_query', 'mean'),
        parameters=('parameters', 'first'),
    )
    agg['std_val_MAE'] = agg['std_val_MAE'].fillna(0.0)
    agg['std_val_support_MAE'] = agg['std_val_support_MAE'].fillna(0.0)
    return agg


def choose_formal(agg: pd.DataFrame, overall_tol: float, tie_rel: float):
    best = float(agg.mean_val_MAE.min())
    d = agg.copy()
    d['within_overall_tolerance'] = d.mean_val_MAE <= best * (1.0 + overall_tol)
    e = d[d.within_overall_tolerance].sort_values('mean_val_support_MAE').copy()
    if len(e) == 0:
        raise ValueError('No formal candidate is within the overall-MAE tolerance.')
    best_support = float(e.iloc[0].mean_val_support_MAE)
    close = e[e.mean_val_support_MAE <= best_support * (1.0 + tie_rel)].copy()
    # When support robustness is effectively tied, prefer the model that is
    # also stronger on directly-masked targets. Treat <=0.5% as another tie.
    best_masked = float(close.mean_val_masked_MAE.min())
    close_masked = close[close.mean_val_masked_MAE <= best_masked * (1.0 + tie_rel)].copy()
    close = close_masked.sort_values([
        'std_val_MAE', 'mean_inference_ms_per_query', 'parameters', 'mean_val_MAE'
    ])
    selected = str(close.iloc[0].model)
    d['selected'] = d.model.astype(str).eq(selected)
    return d, selected, best
