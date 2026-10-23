"""PCFT C3 gain/harm-calibrated choices over Legacy, Propagation and None."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class GainHarmEvolutionPolicy(nn.Module):
    def __init__(self,horizon:int,hidden:int=64,limit:float=3.0):
        super().__init__()
        self.limit=float(limit)
        self.embed=nn.Embedding(horizon,8)
        self.estimator=nn.Sequential(nn.Linear(8+10,hidden),nn.GELU(),nn.Linear(hidden,6))
        self.temperature=nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.estimator[-1].weight)
        nn.init.zeros_(self.estimator[-1].bias)

    def forward(self,base,legacy,propagated,support,availability=None,mode='full',risk_prior=None):
        b,p,n=base.shape
        observed=support['observed'].float();forecast=support['forecast'].float()
        uncertainty=support['uncertainty'].float()
        avail=(availability if availability is not None else (observed+forecast>0)).float()
        h=self.embed(torch.arange(p,device=base.device))[None,:,None].expand(b,p,n,-1)
        feat=torch.cat([h,torch.stack([base.float(),legacy.float(),propagated.float(),
            (legacy-propagated).abs().float(),legacy.abs().float(),propagated.abs().float(),
            observed,forecast,uncertainty,avail],-1)],-1)
        e=self.estimator(feat)
        gain=e[...,:2]
        harm=F.softplus(e[...,2:4])
        uncertainty_action=F.softplus(e[...,4:])
        # Harm is part of the actual decision (unlike v2's detached utility head).
        raw_score=gain-0.4*harm-0.1*uncertainty_action
        # Correctly isolate risk, as opposed to the v2 no_risk flag.
        if mode=='no_risk':raw_score=gain
        if risk_prior is not None and mode!='no_risk':
            raw_score=raw_score+0.02*risk_prior[...,None].float()
        score=torch.cat([raw_score,torch.zeros_like(raw_score[...,:1])],-1)
        score=score.clone()
        score[...,1]=score[...,1].masked_fill(avail<=1e-6,-1e4)
        if mode=='legacy_only':score[...,1]=-1e4
        if mode=='propagation_only':score[...,0]=-1e4
        if mode=='no_future':score[...,:2]=-1e4
        if mode=='no_action_utility':
            # Disable estimated gain/harm, but preserve identical action candidates.
            score=torch.zeros_like(score)
            score[...,1].masked_fill_(avail<=1e-6,-1e4)
        if mode=='fixed_action_policy':
            score=torch.zeros_like(score)+score.new_tensor([1.0,-0.5,0.0])
            score[...,1].masked_fill_(avail<=1e-6,-1e4)
        w=torch.softmax(score/self.temperature.clamp(.5,2.0),dim=-1)
        ld=self.limit*torch.tanh(legacy/self.limit)
        pd=self.limit*torch.tanh(propagated/self.limit)
        correction=w[...,0]*ld+w[...,1]*pd
        return {'correction':correction,'action_weights':w,'action_logits':score,
                'action_expected_gain':gain,'action_expected_harm':harm,
                'action_expected_uncertainty':uncertainty_action,
                'legacy_correction':ld,'propagation_correction':pd}
