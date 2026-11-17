"""Observable-only failure signals and causal historic innovations."""
from __future__ import annotations
import torch
from torch import nn
from corsta.stla import missing_age


def observable_features(x,mask):
    """All features derived strictly from visible history. Output [B,L,N,9]."""
    x=torch.nan_to_num(x.float());m=mask.float().clamp(0,1)
    B,L,N=x.shape
    age=missing_age(m)
    coverage=m.mean(1,keepdim=True).expand(-1,L,-1)
    temporal_coverage=m.mean(2,keepdim=True).expand(-1,-1,N)
    last=torch.zeros((B,N),dtype=x.dtype,device=x.device)
    prev=torch.zeros_like(last)
    last_time=torch.full((B,N),-1,dtype=torch.long,device=x.device)
    prev_time=last_time.clone()
    innovations=[];last_values=[];recent_trends=[];has_prev=[]
    for t in range(L):
        good=m[:,t].bool()
        gap=(t-last_time).clamp_min(1)
        delta=(x[:,t]-last)/gap.to(x.dtype)
        innovation=torch.where(good & (last_time>=0),delta,torch.zeros_like(delta))
        prev=torch.where(good,last,prev)
        prev_time=torch.where(good,last_time,prev_time)
        last=torch.where(good,x[:,t],last)
        last_time=torch.where(good,torch.full_like(last_time,t),last_time)
        last_values.append(last)
        innovations.append(innovation)
        has_prev.append((prev_time>=0).float())
        recent_trends.append(torch.where(last_time>prev_time,(last-prev)/(last_time-prev_time).clamp_min(1),torch.zeros_like(last)))
    return torch.stack([m,age,coverage,temporal_coverage,
                        torch.stack(last_values,1),torch.stack(innovations,1),
                        torch.stack(recent_trends,1),torch.stack(has_prev,1),
                        (m.sum(1,keepdim=True)==0).float().expand(-1,L,-1)],dim=-1)


class FailureEncoder(nn.Module):
    def __init__(self,dim=16):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(9,dim),nn.GELU(),nn.Linear(dim,dim))
    def forward(self,x,mask):
        raw=observable_features(x,mask)
        return {'tokens':self.net(raw),'obs_reliability':mask.float().mean(1),
                'raw':raw,'no_observation':(mask.sum(1)==0)}
