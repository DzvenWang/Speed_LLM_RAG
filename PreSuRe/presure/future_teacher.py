from __future__ import annotations
import numpy as np


def _weighted_mean(err,mask):
    m=mask.astype(bool); num=np.where(m,err,0.0).reshape(err.shape[0],err.shape[1],-1).sum(-1); den=m.reshape(mask.shape[0],mask.shape[1],-1).sum(-1)
    out=num/np.maximum(den,1); out[den<=0]=np.inf; return out


def future_distance(query_y,query_valid,query_direct,candidate_y,candidate_valid,speed_std,level_weight=0.25,evolution_weight=0.75,masked_weight=0.75):
    q=np.asarray(query_y,np.float32)[:,None,:,:]; c=np.asarray(candidate_y,np.float32)
    qv=np.asarray(query_valid,bool)[:,None,:,:]; cv=np.asarray(candidate_valid,bool); direct=np.asarray(query_direct,bool)[:,None,None,:]
    finite=np.isfinite(q)&np.isfinite(c); valid=qv&cv&finite; std=max(float(speed_std),1e-6)
    qd=q-q[:,:,:1,:]; cd=c-c[:,:,:1,:]
    err=float(level_weight)*np.abs(c-q)/std + float(evolution_weight)*np.abs(cd-qd)/std
    overall=_weighted_mean(err,valid); masked=_weighted_mean(err,valid&direct); has=np.isfinite(masked)
    return np.where(has,float(masked_weight)*masked+(1.0-float(masked_weight))*overall,overall).astype(np.float32)


def compute_distance_for_ids(query_split,memory_split,candidate_indices,ids,speed_std,cfg,batch=64):
    ids=np.asarray(ids,np.int64); out=np.empty((len(ids),candidate_indices.shape[1]),np.float32); fc=cfg['future_teacher']
    for a in range(0,len(ids),int(batch)):
        b=min(len(ids),a+int(batch)); qi=ids[a:b]; ci=np.asarray(candidate_indices[qi],np.int64)
        out[a:b]=future_distance(np.asarray(query_split.y[qi],np.float32),np.asarray(query_split.valid[qi],bool),np.asarray(query_split.direct[qi],bool),
            np.asarray(memory_split.y[ci],np.float32),np.asarray(memory_split.valid[ci],bool),speed_std,
            float(fc['level_weight']),float(fc['evolution_weight']),float(fc['masked_weight']))
    return out
