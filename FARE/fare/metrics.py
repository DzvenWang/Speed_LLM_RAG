from __future__ import annotations
import numpy as np


def error_stats(pred,target,valid):
    p=np.asarray(pred); y=np.asarray(target); m=np.asarray(valid,bool)&np.isfinite(p)&np.isfinite(y)
    d=(p-y)[m]
    return {"sum_abs":float(np.abs(d).sum()),"sum_sq":float((d.astype(np.float64)**2).sum()),"count":int(d.size)}

def finalize(s):
    n=max(int(s['count']),1); return {"MAE":float(s['sum_abs']/n),"RMSE":float(np.sqrt(s['sum_sq']/n)),"count":int(s['count'])}

def add_stats(a,b):
    return {"sum_abs":a['sum_abs']+b['sum_abs'],"sum_sq":a['sum_sq']+b['sum_sq'],"count":a['count']+b['count']}

def restoration_mae_raw(restored,full,art,std):
    m=np.asarray(art,bool)
    if not np.any(m): return float('nan')
    return float(np.abs((np.asarray(restored)-np.asarray(full))*float(std))[m].mean())
