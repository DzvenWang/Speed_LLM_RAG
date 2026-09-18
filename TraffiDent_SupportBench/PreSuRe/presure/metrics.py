from __future__ import annotations
from pathlib import Path
import sys, numpy as np

def metric_bundle(cfg,pred,y,valid,direct,support):
    c=Path(cfg['project']['common_root'])
    if str(c) not in sys.path: sys.path.insert(0,str(c))
    from experiment_common.metrics import metric_bundle as f
    return f(pred,y,valid,direct,support)

def per_query_mae_torch(pred,y,valid):
    import torch
    m=valid.bool() & torch.isfinite(pred) & torch.isfinite(y); e=torch.abs(pred-y)*m
    return e.flatten(1).sum(1)/m.flatten(1).sum(1).clamp_min(1)

def weighted_mae_loss(pred,y,valid,direct,support,weights):
    import torch
    def ma(mask):
        m=valid.bool() & mask & torch.isfinite(pred) & torch.isfinite(y)
        if not torch.any(m): return pred.sum()*0
        return torch.abs(pred[m]-y[m]).mean()
    overall=ma(torch.ones_like(valid,dtype=torch.bool))
    dm=direct.bool()[:,None,:].expand_as(valid); sm=support.bool()[:,None,:].expand_as(valid)
    return float(weights['overall_weight'])*overall+float(weights['masked_weight'])*ma(dm)+float(weights['support_weight'])*ma(sm)
