"""C1: separate observed availability, reliability, and horizon-specific support."""
from __future__ import annotations
import torch
from torch import nn


def summarize_history(x,mask,age):
    mask=mask.float()
    d=mask[:,1:]*mask[:,:-1]
    dif=(x[:,1:]-x[:,:-1])*d
    delta=dif.sum(1)/d.sum(1).clamp_min(1.)
    variability=dif.abs().sum(1)/d.sum(1).clamp_min(1.)
    avg=(x*mask).sum(1)/mask.sum(1).clamp_min(1.)
    features=torch.cat((avg,mask.mean(1),1.-age[:,-1],delta,variability),-1)
    return features


class ForecastSupport(nn.Module):
    def __init__(self,d,channels=3,horizon=12,hidden=64):
        super().__init__();self.horizon=horizon
        self.observation=nn.Sequential(nn.Linear(channels*5,hidden),nn.GELU(),nn.Linear(hidden,d))
        self.fuse=nn.Sequential(nn.LayerNorm(d*2),nn.Linear(d*2,d),nn.GELU(),nn.Linear(d,d))
        self.step=nn.Embedding(horizon,d)
        self.forecast=nn.Sequential(nn.Linear(d+2,d),nn.GELU(),nn.Linear(d,d))
        self.quality=nn.Sequential(nn.Linear(d+2,d//2),nn.GELU(),nn.Linear(d//2,1),nn.Sigmoid())
        self.support=nn.Linear(d,1)
    def forward(self,batch,hidden,base_prediction,target_idx):
        x,mask,age=(batch[k] for k in ('x_multi','observed_mask','missing_age'))
        stats=summarize_history(x,mask,age)
        obs=self.observation(stats)
        node=self.fuse(torch.cat((hidden,obs),-1))
        target_mask=mask[...,target_idx]
        cov=target_mask.mean(1)
        freshness=1.-age[:,-1,:,target_idx]
        quality=self.quality(torch.cat((node,cov[...,None],freshness[...,None]),-1))
        B,P,N=base_prediction.shape
        horizon=node[:,None]+self.step.weight[None,:,None,:]
        cond=torch.cat((horizon,base_prediction[...,None],
                        torch.arange(1,P+1,device=node.device)[None,:,None,None].expand(B,-1,N,1).float()/P),-1)
        context=self.forecast(cond)
        score=torch.sigmoid(self.support(context))
        return {'node_context':node,'forecast_context':context,'support_score':score,
                'source_quality':quality,'observed_available':mask.bool()}
