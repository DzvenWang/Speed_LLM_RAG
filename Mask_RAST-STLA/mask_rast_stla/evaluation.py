from __future__ import annotations
import json
from pathlib import Path
import numpy as np

def metrics(pred,true,valid,direct):
    def c(mask):
        m=np.asarray(mask,bool)&np.isfinite(pred)&np.isfinite(true);e=pred[m]-true[m];return (float(np.mean(np.abs(e))) if e.size else float('nan'),float(np.sqrt(np.mean(e*e))) if e.size else float('nan'))
    mae,rmse=c(valid);d=np.asarray(valid,bool)&np.asarray(direct,bool)[:,None,:];mmae,mrmse=c(d);o={'MAE':mae,'RMSE':rmse,'directly_masked_MAE':mmae,'directly_masked_RMSE':mrmse}
    for h in (3,6,12):
        hh=min(h,pred.shape[1]);m=np.asarray(valid[:,hh-1:hh],bool);e=pred[:,hh-1:hh][m]-true[:,hh-1:hh][m];o[f'H{h}_MAE']=float(np.mean(np.abs(e))) if e.size else float('nan')
    return o

def summarize_by_family(pred,true,valid,direct,meta):
    out={'ALL':metrics(pred,true,valid,direct)}
    if 'family' in meta.columns:
        f=meta.family.astype(str).to_numpy()
        for name in sorted(set(f)): ix=np.where(f==name)[0];out[name]=metrics(pred[ix],true[ix],valid[ix],direct[ix])
    return out

def save_summary(path,payload):Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(payload,indent=2,allow_nan=True),encoding='utf-8')
