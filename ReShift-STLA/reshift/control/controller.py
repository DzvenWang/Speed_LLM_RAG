"""Predictability-based residual acceptance, no access to future targets in forward."""
from __future__ import annotations
import torch
from torch import nn


class ResidualBenefitController(nn.Module):
    def __init__(self,cf_dim=32,res_dim=32,hidden=32):
        super().__init__()
        self.acceptance_strength=nn.Parameter(torch.tensor(0.0))
        self.net=nn.Sequential(nn.Linear(cf_dim+res_dim+8,hidden),nn.GELU(),
                               nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias[0:1],0.)
        nn.init.constant_(self.net[-1].bias[1:2],3.)

    def forward(self,p0,p1,p2,cf,res,failure):
        B,P,N=p0.shape
        obs=failure['obs_reliability'][:,None,:].expand(B,P,N)
        spatial=cf['spatial']['confidence']
        spatial_r=res['spatial']['confidence']
        nums=torch.stack([p0,p1,p2,p1-p0,p2-p1,obs,spatial,spatial_r],dim=-1)
        inputs=torch.cat([cf['tokens'],res['tokens'],nums],dim=-1)
        pair=self.net(inputs)
        benefit=pair[...,0]
        proposal=torch.sigmoid(pair[...,1])
        # Strict identity init: controller stage starts exactly from best Joint P2.
        # Gradients can grow acceptance_strength when reductions are useful.
        strength=self.acceptance_strength.clamp(0.0,1.0)
        gate=1.0-strength*(1.0-proposal)
        return {'gate':gate,'benefit_logit':benefit}
