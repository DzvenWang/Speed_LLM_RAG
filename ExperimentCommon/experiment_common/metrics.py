from __future__ import annotations
import numpy as np
from .constants import METRICS_REVISION


def _metric(pred, y, mask, kind='mae'):
    m = np.asarray(mask, bool) & np.isfinite(pred) & np.isfinite(y)
    if not np.any(m):
        return np.nan
    e = np.asarray(pred, np.float64)[m] - np.asarray(y, np.float64)[m]
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
    p, y = _masked_values(pred, true, valid, node_mask)
    if p.size == 0:
        return {'MAE': float('nan'), 'RMSE': float('nan'), 'count': 0}
    e = p - y
    return {'MAE': float(np.mean(np.abs(e))), 'RMSE': float(np.sqrt(np.mean(e * e))), 'count': int(p.size)}


def metric_bundle(pred, true, valid, direct_mask, support_mask):
    """Canonical aggregate metrics used by Baseline/OwnModel evaluation."""
    out = {'metrics_revision': METRICS_REVISION}
    scopes = [('overall', None), ('masked', direct_mask), ('support', support_mask)]
    for name, nm in scopes:
        d = mae_rmse(pred, true, valid, nm)
        out[f'{name}_MAE'] = d['MAE']; out[f'{name}_RMSE'] = d['RMSE']; out[f'{name}_count'] = d['count']
    for h in (3, 6, 12):
        hh = min(h, pred.shape[1]); sl = slice(hh - 1, hh)
        for name, nm in scopes:
            d = mae_rmse(pred[:, sl], true[:, sl], valid[:, sl], nm)
            out[f'H{h}_{name}_MAE'] = d['MAE']; out[f'H{h}_{name}_RMSE'] = d['RMSE']; out[f'H{h}_{name}_count'] = d['count']
        out[f'H{h}_MAE'] = out[f'H{h}_overall_MAE']
        out[f'H{h}_RMSE'] = out[f'H{h}_overall_RMSE']
    return out


def per_query_metrics(pred, y, valid, direct, support):
    """Canonical per-query metrics used by Backbone training/selection."""
    rows = []
    for i in range(len(pred)):
        v = np.asarray(valid[i], bool)
        d = v & np.asarray(direct[i], bool)[None, :]
        s = v & np.asarray(support[i], bool)[None, :]
        r = {
            'MAE': _metric(pred[i], y[i], v, 'mae'),
            'RMSE': _metric(pred[i], y[i], v, 'rmse'),
            'directly_masked_MAE': _metric(pred[i], y[i], d, 'mae'),
            'directly_masked_RMSE': _metric(pred[i], y[i], d, 'rmse'),
            'support_dependent_MAE': _metric(pred[i], y[i], s, 'mae'),
            'support_dependent_RMSE': _metric(pred[i], y[i], s, 'rmse'),
        }
        for h in (3, 6, 12):
            if h <= pred.shape[1]:
                sl = slice(h - 1, h)
                r[f'H{h}_MAE'] = _metric(pred[i, sl], y[i, sl], v[sl], 'mae')
                r[f'H{h}_RMSE'] = _metric(pred[i, sl], y[i, sl], v[sl], 'rmse')
                r[f'H{h}_support_dependent_MAE'] = _metric(pred[i, sl], y[i, sl], s[sl], 'mae')
                r[f'H{h}_support_dependent_RMSE'] = _metric(pred[i, sl], y[i, sl], s[sl], 'rmse')
        rows.append(r)
    return rows


def backbone_metric_summary(pred, true, valid, direct_mask, support_mask):
    """Canonical micro-aggregated metrics using Backbone's historical key names.

    This is the comparison-safe aggregate used by Backbone selection, Baseline,
    and future OwnModel experiments. `per_query_metrics` remains available only
    for diagnostic CSV rows.
    """
    b = metric_bundle(pred, true, valid, direct_mask, support_mask)
    out = {
        'MAE': b['overall_MAE'], 'RMSE': b['overall_RMSE'],
        'directly_masked_MAE': b['masked_MAE'], 'directly_masked_RMSE': b['masked_RMSE'],
        'support_dependent_MAE': b['support_MAE'], 'support_dependent_RMSE': b['support_RMSE'],
    }
    for h in (3, 6, 12):
        out[f'H{h}_MAE'] = b[f'H{h}_overall_MAE']
        out[f'H{h}_RMSE'] = b[f'H{h}_overall_RMSE']
        out[f'H{h}_support_dependent_MAE'] = b[f'H{h}_support_MAE']
        out[f'H{h}_support_dependent_RMSE'] = b[f'H{h}_support_RMSE']
    return out
