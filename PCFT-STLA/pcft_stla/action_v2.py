"""PCFT v2 forecast-horizon action utility: coherent legacy / propagation / none.
Supports missing observations without multiplying action probability by mask coverage.
"""
from __future__ import annotations
import torch
from torch import nn


class ForecastActionController(nn.Module):
    def __init__(self,horizon:int,hidden:int=48,limit:float=3.0):
        super().__init__()
        self.limit=float(limit)
        self.embed=nn.Embedding(int(horizon),8)
        self.predictor=nn.Sequential(nn.Linear(8+8,hidden),nn.GELU(),nn.Linear(hidden,3))
        self.utility=nn.Sequential(nn.Linear(8+8,hidden),nn.GELU(),nn.Linear(hidden,2))
        # Start with a conservative legacy correction but allow propagation
        # and None to be selected when supported by the loss.
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)
        with torch.no_grad(): self.predictor[-1].bias.copy_(torch.tensor((1.0,-0.5,0.0)))
        nn.init.zeros_(self.utility[-1].weight);nn.init.zeros_(self.utility[-1].bias)

    def forward(self,base,legacy,propagated,support,availability=None,mode='full',risk_prior=None):
        # base, legacy and propagated: [B,P,N] in Backbone-normalized units.
        b,p,n=base.shape
        if isinstance(support,dict):
            sobs=support['observed'];sfore=support['forecast'];sunc=support['uncertainty']
        else:
            sobs=support;sfore=torch.zeros_like(sobs);sunc=1-sobs
        emb=self.embed(torch.arange(p,device=base.device))[None,:,None].expand(b,p,n,-1)
        feat=torch.cat([emb,torch.stack([base,legacy,propagated,legacy-propagated,
                  sobs,sfore,sunc,(legacy-propagated).abs()],-1)],-1)
        logits=self.predictor(feat.float())
        if risk_prior is not None:
            logits=logits.clone()
            logits[...,:2]=logits[...,:2]+0.05*risk_prior[...,None]
        if mode in ('no_lag_alignment','legacy_only','no_prop_alignment'):
            logits=logits.clone();logits[...,1]=-1e4
        if mode=='propagation_only':
            logits=logits.clone();logits[...,0]=-1e4
        if mode=='no_future':
            logits=logits.clone();logits[...,:2]=-1e4
        if mode=='no_action_utility':
            logits=torch.zeros_like(logits)+logits.new_tensor([1.0,-0.5,0.0])
        w=torch.softmax(logits,-1)
        limited_legacy=self.limit*torch.tanh(legacy/self.limit)
        limited_prop=self.limit*torch.tanh(propagated/self.limit)
        correction=w[...,0]*limited_legacy+w[...,1]*limited_prop
        expected=self.utility(feat.float())
        return {'correction':correction, 'action_weights':w,'action_logits':logits,
                'action_expected_gain':expected,'legacy_correction':limited_legacy,
                'propagation_correction':limited_prop}
