from __future__ import annotations
import numpy as np


def _masked_values(pred, true, valid, node_mask=None):
    m = np.asarray(valid, bool)
    if node_mask is not None:
        nm = np.asarray(node_mask, bool)
        while nm.ndim < m.ndim:
            nm = np.expand_dims(nm, axis=1)
        m = m & np.broadcast_to(nm, m.shape)
    p = np.asarray(pred, np.float64)[m]
    y = np.asarray(true, np.float64)[m]
    return p, y


def mae_rmse(pred, true, valid, node_mask=None):
    p, y = _masked_values(pred, true, valid, node_mask)
    if p.size == 0:
        return {'MAE': float('nan'), 'RMSE': float('nan'), 'count': 0}
    err = p - y
    return {
        'MAE': float(np.mean(np.abs(err))),
        'RMSE': float(np.sqrt(np.mean(err * err))),
        'count': int(p.size),
    }


def metric_bundle(pred, true, valid, direct_mask, support_mask):
    out = {}
    for name, nm in [('overall', None), ('masked', direct_mask), ('support', support_mask)]:
        d = mae_rmse(pred, true, valid, nm)
        out[f'{name}_MAE'] = d['MAE']; out[f'{name}_RMSE'] = d['RMSE']; out[f'{name}_count'] = d['count']
    for h in (3, 6, 12):
        hh = min(h, pred.shape[1])
        d = mae_rmse(pred[:, hh-1:hh], true[:, hh-1:hh], valid[:, hh-1:hh], None)
        out[f'H{h}_MAE'] = d['MAE']
        ds = mae_rmse(pred[:, hh-1:hh], true[:, hh-1:hh], valid[:, hh-1:hh], support_mask)
        out[f'H{h}_support_MAE'] = ds['MAE']
    return out
