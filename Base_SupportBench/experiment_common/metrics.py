from __future__ import annotations
"""Shared raw-target MAE / RMSE / MAPE evaluators.

MAPE is in percent (not a fraction).  Samples with |ground truth| <= 1.0
in the native target unit are excluded from *MAPE only*, because percentage
error is undefined for zero and unstable for near-zero flow/speed values.
MAE/RMSE retain exactly their former masks and denominators.
Counts/coverage are reported to make the exclusion transparent.
"""
import numpy as np
from .constants import METRICS_REVISION

MAPE_MIN_TARGET_ABS = 1.0
MAPE_PROTOCOL = 'raw_target_percent_exclude_abs_le_1_v1'


def _selection(pred, y, mask):
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.asarray(mask, dtype=bool) & np.isfinite(pred) & np.isfinite(y)
    return pred[m], y[m]


def _metric(pred, y, mask, kind='mae'):
    p, t = _selection(pred, y, mask)
    if kind == 'mape':
        eligible = np.abs(t) > MAPE_MIN_TARGET_ABS
        if not np.any(eligible):
            return float('nan')
        return float(100.0 * np.mean(np.abs(p[eligible] - t[eligible]) / np.abs(t[eligible])))
    if p.size == 0:
        return float('nan')
    e = p - t
    if kind == 'mae':
        return float(np.mean(np.abs(e)))
    if kind == 'rmse':
        return float(np.sqrt(np.mean(e * e)))
    raise KeyError(kind)


def _masked_values(pred, true, valid, node_mask=None):
    m = np.asarray(valid, bool)
    if node_mask is not None:
        nm = np.asarray(node_mask, bool)
        while nm.ndim < m.ndim:
            nm = np.expand_dims(nm, axis=1)
        m = m & np.broadcast_to(nm, m.shape)
    m = m & np.isfinite(pred) & np.isfinite(true)
    return np.asarray(pred, np.float64)[m], np.asarray(true, np.float64)[m]


def mae_rmse(pred, true, valid, node_mask=None):
    """Backwards-compatible API; appends MAPE/count/coverage fields."""
    p, y = _masked_values(pred, true, valid, node_mask)
    n = int(p.size)
    keep = np.abs(y) > MAPE_MIN_TARGET_ABS
    mc = int(np.count_nonzero(keep))
    mape = float(100.0 * np.mean(np.abs(p[keep] - y[keep]) / np.abs(y[keep]))) if mc else float('nan')
    if not n:
        return {'MAE': float('nan'), 'RMSE': float('nan'), 'MAPE': mape,
                'count': 0, 'MAPE_count': 0, 'MAPE_coverage': float('nan')}
    e = p - y
    return {'MAE': float(np.mean(np.abs(e))), 'RMSE': float(np.sqrt(np.mean(e * e))),
            'MAPE': mape, 'count': n, 'MAPE_count': mc, 'MAPE_coverage': float(mc / n)}


def metric_bundle(pred, true, valid, direct_mask):
    """Canonical, micro-aggregated scores for Baseline and Backbone."""
    out = {'metrics_revision': METRICS_REVISION, 'mape_protocol': MAPE_PROTOCOL,
           'mape_min_target_abs': MAPE_MIN_TARGET_ABS}
    scopes = [('overall', None), ('masked', direct_mask)]
    for name, nm in scopes:
        d = mae_rmse(pred, true, valid, nm)
        for key in ('MAE', 'RMSE', 'MAPE', 'count', 'MAPE_count', 'MAPE_coverage'):
            out[f'{name}_{key}'] = d[key]
    for h in (3, 6, 12):
        hh = min(h, pred.shape[1]); sl = slice(hh - 1, hh)
        for name, nm in scopes:
            d = mae_rmse(pred[:, sl], true[:, sl], valid[:, sl], nm)
            for key in ('MAE', 'RMSE', 'MAPE', 'count', 'MAPE_count', 'MAPE_coverage'):
                out[f'H{h}_{name}_{key}'] = d[key]
        for key in ('MAE', 'RMSE', 'MAPE'):
            out[f'H{h}_{key}'] = out[f'H{h}_overall_{key}']
    return out


def per_query_metrics(pred, y, valid, direct):
    """Per-query diagnostics; do not average these rows for headline MAPE."""
    rows = []
    for i in range(len(pred)):
        v = np.asarray(valid[i], bool)
        d = v & np.asarray(direct[i], bool)[None, :]
        r = {}
        for prefix, mask in [('', v), ('directly_masked_', d)]:
            result = mae_rmse(pred[i], y[i], mask)
            for key in ('MAE', 'RMSE', 'MAPE', 'MAPE_count', 'MAPE_coverage'):
                r[f'{prefix}{key}'] = result[key]
        for h in (3, 6, 12):
            if h <= pred.shape[1]:
                sl = slice(h - 1, h)
                result = mae_rmse(pred[i, sl], y[i, sl], v[sl])
                for key in ('MAE', 'RMSE', 'MAPE', 'MAPE_count'):
                    r[f'H{h}_{key}'] = result[key]
        rows.append(r)
    return rows


def backbone_metric_summary(pred, true, valid, direct_mask):
    """Micro-aggregate with historical Backbone keys plus additive MAPE keys."""
    b = metric_bundle(pred, true, valid, direct_mask)
    out = {}
    for key in ('MAE', 'RMSE', 'MAPE'):
        out[key] = b[f'overall_{key}']
        out[f'directly_masked_{key}'] = b[f'masked_{key}']
    for key in ('count', 'MAPE_count', 'MAPE_coverage'):
        out[key] = b[f'overall_{key}']
        out[f'directly_masked_{key}'] = b[f'masked_{key}']
    for h in (3, 6, 12):
        for key in ('MAE', 'RMSE', 'MAPE', 'MAPE_count'):
            out[f'H{h}_{key}'] = b[f'H{h}_overall_{key}']
    return out
