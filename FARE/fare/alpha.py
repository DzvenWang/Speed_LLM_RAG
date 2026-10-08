"""Validation-only scalar alpha sweeps for forecasting correction proposals.

A sweep is a diagnostic / calibration technique. It is NOT evidence of
performance on a held-out test and must not select alpha on test data.
"""
from __future__ import annotations
import csv
import numpy as np
from pathlib import Path
from .config import json_write

DEFAULT_ALPHA=(0.,.025,.05,.075,.10,.12,.15,.20,.25,.30,.40,.50,.60)
DEFAULT_MULTIPLIERS=(0.,.25,.5,.75,1.,1.25,1.5,2.,2.5,3.,4.)

def scalar_sweep(base,candidate,y,valid,alphas=DEFAULT_ALPHA,score_rmse_weight=.05,max_alpha=.6):
    base=np.asarray(base,dtype=np.float64);candidate=np.asarray(candidate,dtype=np.float64)
    y=np.asarray(y,dtype=np.float64);valid=np.asarray(valid,dtype=bool)
    if base.shape!=candidate.shape or base.shape!=y.shape or base.shape!=valid.shape:
        raise ValueError('base/candidate/target/valid shape mismatch')
    if not valid.any():raise ValueError('No valid supervised elements')
    if not (np.isfinite(base[valid]).all() and np.isfinite(candidate[valid]).all() and np.isfinite(y[valid]).all()):
        raise FloatingPointError('Nonfinite validation forecast')
    rows=[]
    for a in alphas:
        a=float(a)
        if not (0<=a<=max_alpha):raise ValueError(f'alpha={a} outside 0..{max_alpha}')
        error=(base+a*(candidate-base)-y)[valid]
        mae=float(np.abs(error).mean());rmse=float(np.sqrt((error**2).mean()))
        rows.append({'alpha':a,'MAE':mae,'RMSE':rmse,'objective':mae+score_rmse_weight*rmse})
    best=min(rows,key=lambda row:(row['objective'],row['alpha']))
    return {'grid':rows,'best':best,'selection_split':'val',
            'criterion':f'MAE+{score_rmse_weight}*RMSE','valid_positions':int(valid.sum())}

def save_sweep(result,output_dir):
    path=Path(output_dir);path.mkdir(parents=True,exist_ok=True)
    json_write(path/'alpha_sweep.json',result)
    with (path/'alpha_sweep.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=('alpha','MAE','RMSE','objective'))
        writer.writeheader();writer.writerows(result['grid'])
    return path
