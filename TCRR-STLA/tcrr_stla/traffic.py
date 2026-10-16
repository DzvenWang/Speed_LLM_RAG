"""Observed-only traffic signatures; no query future labels are consumed.

All features operate on normalized speed/flow in the Backbone protocol.  A
regime is deliberately a soft statistical *traffic state*, not a claim of
physically measured congestion in flow-only data.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


def node_features(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return [B,N,8] features; robust to fully missing nodes and short history."""
    if x.shape != mask.shape or x.ndim != 3:
        raise ValueError('x and observed mask must be [B,H,N]')
    with torch.autocast(device_type=x.device.type, enabled=False):
        x=torch.nan_to_num(x.float())
        m=(mask.float()>0.5).float()
        b,h,n=x.shape
        count=m.sum(1);cover=count/h
        mean=(x*m).sum(1)/count.clamp_min(1.)
        var=(((x-mean[:,None])**2)*m).sum(1)/count.clamp_min(1.)
        index=torch.arange(h,device=x.device)[None,:,None].expand(b,h,n)
        fi=torch.where(m.bool(),index,h).min(1).values.clamp(max=h-1)
        li=torch.where(m.bool(),index,-1).max(1).values.clamp(min=0)
        first=torch.gather(x,1,fi[:,None]).squeeze(1)
        last=torch.gather(x,1,li[:,None]).squeeze(1)
        delta=(last-first)/(li-fi).clamp_min(1.).float()
        missing_age=(h-1-li).float()/max(h-1,1)
        recent=m[:,-max(1,h//3):].mean(1)
        motion=(x[:,-1]-x[:,0])/(max(h-1,1))
        valid=(count>0).float()
        f=torch.stack([mean,last,delta,torch.sqrt(var+1e-6),cover,
                       missing_age,recent,motion],-1)
        f[...,:4]=f[...,:4]*valid[...,None]
        f[...,7]=f[...,7]*m[:,-1]*m[:,0]
        return torch.nan_to_num(f).clamp(-20,20)


def global_signature(x: torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
    """[B,12], scale-aware, mask-aware, independent of network node count."""
    f=node_features(x,mask)
    weighted=f[...,4].clamp_min(0.)
    den=weighted.sum(1).clamp_min(1.)
    parts=[(f[...,i]*weighted).sum(1)/den for i in range(8)]
    # Network state distribution distinguishes spatially heterogeneous failures.
    obs=f[...,4]>0
    for i in (0,2,3,5):
        v=f[...,i];mu=(v*weighted).sum(1)/den
        parts.append(torch.sqrt(((v-mu[:,None]).square()*weighted).sum(1)/den+1e-6))
    return torch.stack(parts,-1).float()


class TrafficResidualHead(nn.Module):
    """Zero-initialized additive traffic reranking and residual component control."""
    def __init__(self, hidden=48):
        super().__init__()
        self.node=nn.Sequential(nn.Linear(8,hidden),nn.GELU(),nn.Linear(hidden,16))
        self.regime=nn.Sequential(nn.Linear(8,hidden),nn.GELU(),nn.Linear(hidden,4))
        self.match=nn.Sequential(nn.Linear(16*3+6,hidden),nn.GELU(),nn.Linear(hidden,4))
        # outputs logit_bias, normal_scale_bias, failure_scale_bias, utility forecast
        nn.init.zeros_(self.match[-1].weight)
        nn.init.zeros_(self.match[-1].bias)
        self.mix=nn.Sequential(nn.Linear(8+3,hidden),nn.GELU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.mix[-1].weight)
        nn.init.zeros_(self.mix[-1].bias)

    def forward(self,x,mask,candidate_history):
        q=node_features(x,mask)
        b,k,h,n=candidate_history.shape
        m=mask[:,None].expand(b,k,h,n)
        c=node_features(candidate_history.reshape(b*k,h,n),m.reshape(b*k,h,n)).reshape(b,k,n,8)
        qz=self.node(q.float());cz=self.node(c.float())
        qz=qz[:,None].expand_as(cz)
        rq=torch.softmax(self.regime(q),-1)
        rc=torch.softmax(self.regime(c),-1)
        sim=(rq[:,None]*rc).sum(-1)
        entropy_q=-(rq.clamp_min(1e-7)*rq.clamp_min(1e-7).log()).sum(-1)
        entropy_c=-(rc.clamp_min(1e-7)*rc.clamp_min(1e-7).log()).sum(-1)
        extra=torch.stack([q[...,4][:,None].expand(b,k,n),
                           c[...,4],(q[...,2][:,None]-c[...,2]).abs(),
                           sim,entropy_q[:,None].expand(b,k,n),entropy_c],-1)
        pair=torch.cat([qz,cz,(qz-cz).abs(),extra],-1)
        o=self.match(pair)
        return dict(logit_bias=o[...,0],normal_bias=o[...,1],failure_bias=o[...,2],
                    expected_gain=o[...,3], query_features=q,regime_probability=rq,
                    regime_entropy=entropy_q)

    def mixture_gain(self,qfeatures,base,proposal,cover):
        b,p,n=base.shape
        f=torch.cat([qfeatures[:,None].expand(b,p,n,8),
                     base[...,None],proposal[...,None],cover[...,None]],-1)
        return self.mix(f).squeeze(-1)
