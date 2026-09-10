from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import metric_bundle


def load_prediction_file(path):
    z=np.load(path,allow_pickle=False)
    if 'view_id' not in z or 'pred' not in z: raise ValueError('Prediction NPZ must contain view_id and pred.')
    return z['view_id'].astype(str),np.asarray(z['pred'],np.float32)


def evaluate_prediction_file(cache_root, split, pred_path):
    r=Path(cache_root); meta=pd.read_csv(r/f'{split}_manifest.csv'); vids,pred=load_prediction_file(pred_path)
    order={v:i for i,v in enumerate(vids)}
    missing=[v for v in meta.view_id.astype(str) if v not in order]
    if missing: raise ValueError(f'{len(missing)} manifest view_ids missing from prediction file; first={missing[:3]}')
    idx=np.asarray([order[v] for v in meta.view_id.astype(str)],np.int64); pred=pred[idx]
    true=np.load(r/f'{split}_y_raw.npy',mmap_mode='r'); valid=np.load(r/f'{split}_future_valid.npy',mmap_mode='r'); direct=np.load(r/f'{split}_direct_mask.npy',mmap_mode='r'); support=np.load(r/f'{split}_support_mask.npy',mmap_mode='r')
    if pred.shape!=true.shape: raise ValueError(f'Prediction shape {pred.shape} != target {true.shape}')
    overall=metric_bundle(pred,true,valid,direct,support); rows=[{'family':'ALL',**overall}]
    for f,g in meta.groupby('family',sort=False):
        ii=g.index.to_numpy(np.int64); rows.append({'family':str(f),**metric_bundle(pred[ii],true[ii],valid[ii],direct[ii],support[ii])})
    return pd.DataFrame(rows)
