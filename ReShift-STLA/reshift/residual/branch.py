"""Forecast residual after CF compensation, not another failure-response predictor."""
from __future__ import annotations
import torch
from torch import nn
from reshift.counterfactual.branch import CrossTime


class ResidualBranch(nn.Module):
    def __init__(self,hidden=64,failure=16,cf_dim=32,dim=32,horizon=12,heads=4,scale=.35):
        super().__init__()
        self.pos=nn.Embedding(horizon,dim)
        self.query=nn.Sequential(nn.Linear(3+cf_dim,dim),nn.GELU(),nn.LayerNorm(dim))
        self.history_cross=CrossTime(dim,hidden+failure+2,dim,heads)
        self.spatial_fusion=nn.Sequential(nn.Linear(dim*2,dim),nn.GELU(),nn.Linear(dim,dim))
        self.reverse_cross=CrossTime(hidden,dim,hidden,heads)
        self.head=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        nn.init.normal_(self.head[-1].weight,mean=0,std=1e-3);nn.init.zeros_(self.head[-1].bias)
        self.uncertainty=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,1))
        self.scale=float(scale)

    def temporal_tokens(self,h1,p0,p1,cf_tokens,failure):
        B,L,N,D=h1.shape;P=p1.shape[1]
        f=failure['tokens'];raw=failure['raw']
        innovation=raw[...,5:6];trend=raw[...,6:7]
        c=torch.stack([p0,p1,p1-p0],dim=-1)
        q=self.query(torch.cat([c,cf_tokens],dim=-1))
        q=q+self.pos(torch.arange(P,device=h1.device))[None,:,None,:]
        hist=torch.cat([h1,f,innovation,trend],dim=-1)
        return self.history_cross(q,hist)

    def forward(self,h1,p0,p1,cf_tokens,failure,spatial,temporal=None):
        temp=temporal if temporal is not None else self.temporal_tokens(h1,p0,p1,cf_tokens,failure)
        token=self.spatial_fusion(torch.cat([temp,spatial['tokens']],dim=-1))
        delta=self.head(self.reverse_cross(h1,token))*self.scale
        sigma=torch.nn.functional.softplus(self.uncertainty(token).squeeze(-1))+1e-4
        return {'tokens':token,'delta_h':delta,'uncertainty':sigma,'spatial':spatial}
