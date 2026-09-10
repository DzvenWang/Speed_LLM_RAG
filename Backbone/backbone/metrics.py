from __future__ import annotations
import numpy as np

METRICS_REVISION = 'rmse_fix1'


def _metric(pred, y, mask, kind='mae'):
    m = np.asarray(mask, bool) & np.isfinite(pred) & np.isfinite(y)
    if not np.any(m):
        return np.nan
    e = np.asarray(pred)[m] - np.asarray(y)[m]
    if kind == 'mae':
        return float(np.mean(np.abs(e)))
    if kind == 'rmse':
        return float(np.sqrt(np.mean(e * e)))
    raise KeyError(kind)


def per_query_metrics(pred, y, valid, direct, support):
    rows = []
    for i in range(len(pred)):
        v = valid[i]
        d = v & direct[i][None, :]
        s = v & support[i][None, :]
        r = {
            'MAE': _metric(pred[i], y[i], v, 'mae'),
            'RMSE': _metric(pred[i], y[i], v, 'rmse'),
            'directly_masked_MAE': _metric(pred[i], y[i], d, 'mae'),
            'directly_masked_RMSE': _metric(pred[i], y[i], d, 'rmse'),
            'support_dependent_MAE': _metric(pred[i], y[i], s, 'mae'),
            'support_dependent_RMSE': _metric(pred[i], y[i], s, 'rmse'),
        }
        for h in [3, 6, 12]:
            if h <= pred.shape[1]:
                r[f'H{h}_MAE'] = _metric(pred[i, h-1:h], y[i, h-1:h], v[h-1:h], 'mae')
                r[f'H{h}_RMSE'] = _metric(pred[i, h-1:h], y[i, h-1:h], v[h-1:h], 'rmse')
                r[f'H{h}_support_dependent_MAE'] = _metric(pred[i, h-1:h], y[i, h-1:h], s[h-1:h], 'mae')
                r[f'H{h}_support_dependent_RMSE'] = _metric(pred[i, h-1:h], y[i, h-1:h], s[h-1:h], 'rmse')
        rows.append(r)
    return rows
