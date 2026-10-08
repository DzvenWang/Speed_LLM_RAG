from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class Reliability(nn.Module):
    def __init__(self,hidden=32):
        super().__init__();self.net=nn.Sequential(nn.Linear(6,hidden),nn.GELU(),nn.Linear(hidden,1))
        nn.init.constant_(self.net[-1].bias,0.)
    def forward(self,batch,ctx,retrieval,state,innovation,regions):
        b,h,n=batch['x_obs'].shape;owner=regions.owner
        coverage=ctx['coverage'][:,owner][:,None,:].expand(b,h,n)
        missing=ctx['missing'][:,owner][:,None,:].expand(b,h,n)
        w=retrieval['weights'][:,owner,:]
        entropy=-(w*w.clamp_min(1e-9).log()).sum(-1)[:,None,:].expand(b,h,n)
        valid=retrieval['group_valid'][:,owner][:,None,:].expand(b,h,n)
        f=torch.stack([coverage,missing,entropy,state['uncertainty'],innovation['uncertainty'],batch['obs_mask'].float()],-1)
        c=torch.sigmoid(self.net(f)).squeeze(-1)*valid.float()
        return c

class UtilityRiskController(nn.Module):
    def __init__(self,hidden=64,horizon=12):
        super().__init__();self.horizon=horizon
        self.feature=nn.Sequential(nn.Linear(7,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        self.utility=nn.Linear(hidden,1);self.risk=nn.Linear(hidden,1)
        self.policy=nn.Sequential(nn.Linear(hidden+2,hidden),nn.GELU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.policy[-1].weight);nn.init.constant_(self.policy[-1].bias,-2.0)
        self.register_buffer('temperature',torch.tensor(1.))
        self.register_buffer('offset',torch.tensor(0.))
    def from_raw(self,f):
        hidden=self.feature(f[...,:7])
        util=self.utility(hidden).squeeze(-1)
        risk=F.softplus(self.risk(hidden)).squeeze(-1)
        frac=f[...,7]
        inp=torch.cat([hidden,frac[...,None],(util-risk)[...,None]],-1)
        gate=torch.sigmoid((self.policy(inp).squeeze(-1)+self.offset)/self.temperature.clamp_min(.05))
        return util,risk,gate
    def forward(self,base,candidate,conf,innovation,state,retrieval,regions):
        b,p,n=base.shape;owner=regions.owner
        confp=conf.mean(1)[:,None,:].expand(b,p,n)
        unc=state['uncertainty'].mean(1)[:,None,:].expand(b,p,n)
        iunc=innovation['uncertainty'].mean(1)[:,None,:].expand(b,p,n)
        delta=candidate-base
        score=retrieval['weights'][:,owner,:].max(-1).values[:,None,:].expand(b,p,n)
        frac=torch.arange(p,device=base.device,dtype=base.dtype).view(1,p,1).expand(b,p,n)/max(1,p-1)
        f=torch.stack([base,candidate,delta,confp,unc,iunc,score],-1)
        hidden=self.feature(f)
        util=self.utility(hidden).squeeze(-1)
        risk=F.softplus(self.risk(hidden)).squeeze(-1)
        inp=torch.cat([hidden,frac[...,None],(util-risk)[...,None]],-1)
        gate=torch.sigmoid((self.policy(inp).squeeze(-1)+self.offset)/self.temperature.clamp_min(.05))
        valid=retrieval['group_valid'][:,owner][:,None,:].expand(b,p,n)
        gate=gate*valid.float()
        pred=base+gate*delta
        return {'prediction':pred,'gate':gate,'utility':util,'risk':risk,'features':hidden,'raw_features':torch.cat([f,frac[...,None]],-1)}
