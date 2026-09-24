from __future__ import annotations
import torch


def weighted_mae(pred,y,valid,direct,support,weights):
    valid=valid.bool(); direct=direct.bool()[:,None,:].expand_as(valid); support=support.bool()[:,None,:].expand_as(valid)
    finite=torch.isfinite(pred)&torch.isfinite(y)
    def mae(mask):
        m=valid&mask&finite
        return torch.abs(pred[m]-y[m]).mean() if torch.any(m) else pred.sum()*0.0
    overall=mae(torch.ones_like(valid,dtype=torch.bool)); masked=mae(direct); supp=mae(support)
    loss=float(weights['overall_weight'])*overall+float(weights['masked_weight'])*masked+float(weights['support_weight'])*supp
    return loss,{'overall':overall.detach(),'masked':masked.detach(),'support':supp.detach()}
