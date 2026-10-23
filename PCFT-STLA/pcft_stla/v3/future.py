"""Support-anchored Future innovation; no query-future inputs at inference."""
from __future__ import annotations
import torch
from torch import nn


class SupportAnchoredFuture(nn.Module):
    """Learn a support-/age-/horizon-dependent amplitude for historical innovations.

    The output is exclusively a multiplicative transport of observed Train future
    evidence, never an unconstrained future prediction head.
    """
    def __init__(self,horizon,hidden=32):
        super().__init__()
        self.embed=nn.Embedding(horizon,8)
        self.scale=nn.Sequential(nn.Linear(8+4,hidden),nn.GELU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.scale[-1].weight)
        nn.init.zeros_(self.scale[-1].bias)

    def forward(self,residual,mask,neighbour_coverage,return_details=False):
        b,c,p,n=residual.shape
        cov=mask.float().mean(1)
        age=mask.flip(1).argmax(1).float()/max(1,mask.shape[1]-1)
        age=torch.where(mask.bool().any(1),age,torch.ones_like(age))
        recent=mask[:,-max(1,mask.shape[1]//3):].mean(1)
        emb=self.embed(torch.arange(p,device=residual.device))[None,:,None].expand(b,p,n,-1)
        f=torch.cat([emb,torch.stack([cov,neighbour_coverage,age,recent],-1)[:,None].expand(b,p,n,4)],-1)
        s=2*torch.sigmoid(self.scale(f.float()).squeeze(-1))
        transported=residual*s[:,None]
        return (transported,s) if return_details else transported
