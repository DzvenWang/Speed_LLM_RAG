from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

class Reliability(nn.Module):
    """Only observed/retrieved inference-time data, never full query ground truth."""
    def __init__(self,hidden=32):
        super().__init__();self.net=nn.Sequential(nn.Linear(7,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.obs_reliability=nn.Linear(hidden,1)
        nn.init.normal_(self.obs_reliability.weight,std=0.003)
        nn.init.zeros_(self.obs_reliability.bias)
        nn.init.constant_(self.net[-1].bias,0.)
    def forward(self,batch,ctx,retrieval,state,innovation,regions,observability=None):
        b,h,n=batch['x_obs'].shape;owner=regions.owner
        coverage=ctx['coverage'][:,owner][:,None,:].expand(b,h,n)
        missing=ctx['missing'][:,owner][:,None,:].expand(b,h,n)
        w=retrieval['weights'][:,owner,:]
        entropy=-(w*w.clamp_min(1e-9).log()).sum(-1)[:,None,:].expand(b,h,n)
        valid=retrieval['group_valid'][:,owner][:,None,:].expand(b,h,n)
        if observability is None:
            obs_score=torch.zeros_like(coverage)
        else:
            obs_score=torch.tanh(self.obs_reliability(observability).squeeze(-1))[:,owner][:,None,:].expand(b,h,n)
        f=torch.stack([coverage,missing,entropy,state['uncertainty'],innovation['uncertainty'],batch['obs_mask'].float(),obs_score],-1)
        return torch.sigmoid(self.net(f)).squeeze(-1)*valid.float()

class UtilityRiskController(nn.Module):
    """Bounded, horizon-aware correction policy. Shared architecture across speed/flow/nets.

    The candidate is a *proposal*, not a forecast that must be trusted at alpha=1.
    The policy is differentiable under forecast loss and separately trainable
    from purged OOF utility/risk labels. All control features are inference-safe.
    """
    def __init__(self,hidden=64,horizon=12,max_alpha=.6,initial_alpha=.12):
        super().__init__();self.horizon=int(horizon)
        self.max_alpha=float(max_alpha)
        if not (0 < initial_alpha < max_alpha <= 1):
            raise ValueError('Require 0 < initial_alpha < max_alpha <= 1')
        self.feature=nn.Sequential(nn.Linear(7,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        self.utility=nn.Linear(hidden,1);self.risk=nn.Linear(hidden,1)
        # Horizon feature, learned OOF utility/risk and retrieval-sensitive hidden state.
        self.policy=nn.Sequential(nn.Linear(hidden+2,hidden),nn.GELU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.policy[-1].weight)
        nn.init.zeros_(self.policy[-1].bias)
        self.horizon_bias=nn.Parameter(torch.zeros(horizon))
        self.register_buffer('prior_logit',torch.tensor(math.log(initial_alpha/(max_alpha-initial_alpha)),dtype=torch.float32))
        self.register_buffer('temperature',torch.tensor(1.))
        self.register_buffer('offset',torch.tensor(0.))
    def _predict(self,features,frac,ignore_risk=False,ignore_utility=False):
        hidden=self.feature(features)
        util=self.utility(hidden).squeeze(-1)
        risk=F.softplus(self.risk(hidden)).squeeze(-1)
        decision_util=torch.zeros_like(util) if ignore_utility else util
        decision_risk=torch.zeros_like(risk) if ignore_risk else risk
        inp=torch.cat([hidden,frac[...,None],(decision_util-decision_risk)[...,None]],-1)
        # Fractional time step determines an interpolation of horizon bias for
        # OOF batches and avoids relying on fixed batch dimensions.
        idx=(frac.clamp(0,1)*(self.horizon-1)).round().long()
        logit=(self.policy(inp).squeeze(-1)+self.prior_logit+
               self.horizon_bias[idx]+self.offset)/self.temperature.clamp_min(.05)
        gate=self.max_alpha*torch.sigmoid(logit)
        return util,risk,gate
    def from_raw(self,raw):
        if raw.size(-1)!=8:raise ValueError(f'Expected 8 OOF features, got {raw.size(-1)}')
        return self._predict(raw[...,:7],raw[...,7])
    def forward(self,base,candidate,conf,innovation,state,retrieval,regions,ignore_risk=False,ignore_utility=False,constant_alpha=None):
        b,p,n=base.shape;owner=regions.owner
        confp=conf.mean(1)[:,None,:].expand(b,p,n)
        unc=state['uncertainty'].mean(1)[:,None,:].expand(b,p,n)
        iunc=innovation['uncertainty'].mean(1)[:,None,:].expand(b,p,n)
        delta=candidate-base
        score=retrieval['weights'][:,owner,:].max(-1).values[:,None,:].expand(b,p,n)
        frac=torch.arange(p,device=base.device,dtype=base.dtype).view(1,p,1).expand(b,p,n)/max(1,p-1)
        f=torch.stack([base,candidate,delta,confp,unc,iunc,score],-1)
        util,risk,gate=self._predict(f,frac,ignore_risk,ignore_utility)
        if constant_alpha is not None:
            alpha=float(constant_alpha)
            if not 0 <= alpha <= self.max_alpha:raise ValueError('Invalid constant alpha')
            gate=torch.full_like(gate,alpha)
        valid=retrieval['group_valid'][:,owner][:,None,:].expand(b,p,n)
        gate=gate*valid.float()
        pred=base+gate*delta
        return {'prediction':pred,'gate':gate,'utility':util,'risk':risk,
                'raw_features':torch.cat([f,frac[...,None]],-1)}
