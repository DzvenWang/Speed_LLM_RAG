from __future__ import annotations
import numpy as np
import torch


def _masked_mean_np(x,mask):
    m=np.asarray(mask,bool)&np.isfinite(x)
    return float(np.asarray(x,np.float64)[m].mean()) if np.any(m) else float('nan')


def metric_dict_np(pred,y,valid,direct,support):
    pred=np.asarray(pred,np.float32); y=np.asarray(y,np.float32); valid=np.asarray(valid,bool)
    direct=np.asarray(direct,bool)[:,None,:]; support=np.asarray(support,bool)[:,None,:]
    err=np.abs(pred-y); sq=(pred-y)**2; out={}
    for name,m in {'overall':valid,'masked':valid&direct,'support':valid&support}.items():
        out[f'{name}_mae']=_masked_mean_np(err,m); mse=_masked_mean_np(sq,m); out[f'{name}_rmse']=float(np.sqrt(mse)) if np.isfinite(mse) else float('nan'); out[f'{name}_count']=int(m.sum())
    return out


def horizon_masked_mae(pred,y,valid,direct):
    p=np.asarray(pred,np.float32); yy=np.asarray(y,np.float32); vv=np.asarray(valid,bool); dd=np.asarray(direct,bool)[:,None,:]
    out=[]
    for h in range(p.shape[1]): out.append(_masked_mean_np(np.abs(p[:,h]-yy[:,h]),vv[:,h]&dd[:,0]))
    return out


def weighted_forecast_loss(pred,y,valid,direct,support,weights):
    valid=valid.bool(); direct=direct.bool()[:,None,:].expand_as(valid); support=support.bool()[:,None,:].expand_as(valid); finite=torch.isfinite(pred)&torch.isfinite(y)
    def mae(mask):
        m=valid&mask&finite
        return torch.abs(pred[m]-y[m]).mean() if torch.any(m) else pred.sum()*0.0
    overall=mae(torch.ones_like(valid,dtype=torch.bool)); masked=mae(direct); supp=mae(support)
    total=float(weights['overall_weight'])*overall+float(weights['masked_weight'])*masked+float(weights['support_weight'])*supp
    return total,{'overall':overall.detach(),'masked':masked.detach(),'support':supp.detach()}


def improvement_pct(base,model):
    if not (np.isfinite(base) and np.isfinite(model)) or base==0: return float('nan')
    return float((base-model)/base*100.0)
