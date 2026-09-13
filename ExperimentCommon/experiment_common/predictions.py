from __future__ import annotations
from pathlib import Path
import numpy as np
from .constants import PREDICTION_CONTRACT_REVISION
from .utils import write_json, sha256_file

REQUIRED_KEYS = ('view_id', 'pred')


def as_unicode_array(values):
    values = [str(x) for x in values]
    width = max([1] + [len(x) for x in values])
    return np.asarray(values, dtype=f'<U{width}')


def save_prediction_npz(path, view_id, pred, *, manifest_path=None, metadata=None):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    vids = as_unicode_array(view_id)
    arr = np.asarray(pred, np.float32)
    if arr.ndim != 3:
        raise ValueError(f'pred must be [M,P,N], got {arr.shape}')
    if len(vids) != arr.shape[0]:
        raise ValueError('view_id/pred row mismatch')
    if not np.isfinite(arr).all():
        raise ValueError('pred contains non-finite values')
    np.savez_compressed(p, view_id=vids, pred=arr)
    if manifest_path is not None:
        obj = {
            'prediction_contract_revision': PREDICTION_CONTRACT_REVISION,
            'prediction_file': str(p),
            'prediction_sha256': sha256_file(p),
            'rows': int(arr.shape[0]), 'horizon': int(arr.shape[1]), 'nodes': int(arr.shape[2]),
            **(metadata or {}),
        }
        write_json(manifest_path, obj)
    return p


def load_prediction_npz(path, *, expected_rows=None, expected_horizon=None, expected_nodes=None):
    p = Path(path)
    z = np.load(p, allow_pickle=False)
    missing = [k for k in REQUIRED_KEYS if k not in z]
    if missing:
        raise ValueError(f'Prediction NPZ missing keys {missing}: {p}')
    vids = z['view_id'].astype(str)
    pred = np.asarray(z['pred'], np.float32)
    if pred.ndim != 3:
        raise ValueError(f'Prediction shape must be [M,P,N], got {pred.shape}')
    if len(vids) != pred.shape[0]:
        raise ValueError('Prediction view_id/pred row mismatch')
    if expected_rows is not None and pred.shape[0] != int(expected_rows):
        raise ValueError(f'Prediction rows {pred.shape[0]} != {expected_rows}')
    if expected_horizon is not None and pred.shape[1] != int(expected_horizon):
        raise ValueError(f'Prediction horizon {pred.shape[1]} != {expected_horizon}')
    if expected_nodes is not None and pred.shape[2] != int(expected_nodes):
        raise ValueError(f'Prediction nodes {pred.shape[2]} != {expected_nodes}')
    if not np.isfinite(pred).all():
        raise ValueError('Prediction contains non-finite values')
    return vids, pred
