from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import metric_bundle
import sys
COMMON=(next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd())).parent / 'Base_SupportBench'
if not COMMON.exists(): COMMON=Path(__file__).resolve().parents[2]/'Base_SupportBench'
if str(COMMON) not in sys.path: sys.path.insert(0,str(COMMON))
from experiment_common.predictions import load_prediction_npz


def load_prediction_file(path):
    return load_prediction_npz(path)


def evaluate_prediction_file(cache_root, split, pred_path):
    r=Path(cache_root); meta=pd.read_csv(r/f'{split}_manifest.csv'); vids,pred=load_prediction_file(pred_path)
    order={v:i for i,v in enumerate(vids)}
    missing=[v for v in meta.view_id.astype(str) if v not in order]
    if missing: raise ValueError(f'{len(missing)} manifest view_ids missing from prediction file; first={missing[:3]}')
    idx=np.asarray([order[v] for v in meta.view_id.astype(str)],np.int64); pred=pred[idx]
    true=np.load(r/f'{split}_y_raw.npy',mmap_mode='r'); valid=np.load(r/f'{split}_future_valid.npy',mmap_mode='r'); direct=np.load(r/f'{split}_direct_mask.npy',mmap_mode='r')
    if pred.shape!=true.shape: raise ValueError(f'Prediction shape {pred.shape} != target {true.shape}')
    overall=metric_bundle(pred,true,valid,direct); rows=[{'family':'ALL',**overall}]
    for f,g in meta.groupby('family',sort=False):
        ii=g.index.to_numpy(np.int64); rows.append({'family':str(f),**metric_bundle(pred[ii],true[ii],valid[ii],direct[ii])})
    return pd.DataFrame(rows)
