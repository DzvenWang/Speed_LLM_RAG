from __future__ import annotations
import numpy as np
from .evaluate import predict_raw
from .metrics import masked_mae
from .transport import apply_transport


def baseline_mae(adapter,ck,q,device):
    pred=predict_raw(adapter,ck,q["x_obs"],q["tod_hist"],q["dow_hist"],device)[0]
    return float(masked_mae(pred,q["y_raw"],q["future_valid"],q["direct_mask"])[0]),pred


def candidate_batch(adapter,ck,q,candidates,transport,cfg,device):
    xs=[]; fill_counts=[]
    for c in candidates:
        xr,f=apply_transport(transport,q,c,cfg); xs.append(xr); fill_counts.append(int(f.sum()))
    if not xs: return np.empty((0,),np.float32),np.empty((0,),np.int32)
    X=np.stack(xs); tod=np.repeat(q["tod_hist"][None,:],len(xs),axis=0); dow=np.repeat(q["dow_hist"][None,:],len(xs),axis=0)
    pred=predict_raw(adapter,ck,X,tod,dow,device)
    y=np.repeat(q["y_raw"][None,:,:],len(xs),axis=0); v=np.repeat(q["future_valid"][None,:,:],len(xs),axis=0); d=np.repeat(q["direct_mask"][None,:],len(xs),axis=0)
    return masked_mae(pred,y,v,d).astype(np.float32),np.asarray(fill_counts,np.int32)
