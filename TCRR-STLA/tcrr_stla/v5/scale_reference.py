"""C3: scale utility controller with exact zero option and validation calibration.

Calibration is fitted on validation only; it never sees test futures. It is a
selective-control heuristic, not a distribution-free coverage guarantee.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class CorrectionScaleController(nn.Module):
    def __init__(self,latent,hidden=64,scales=(0.,.25,.5,.75,1.)):
        super().__init__()
        if not scales or scales[0]!=0 or any(x<0 or x>1 for x in scales):
            raise ValueError('scales must start at 0 and remain in [0,1]')
        self.register_buffer('scales',torch.tensor(scales,dtype=torch.float32))
        self.horizon_embed=nn.Embedding(512,8)
        self.net=nn.Sequential(nn.Linear(latent+8+9,hidden),nn.GELU(),
                               nn.Linear(hidden,hidden),nn.GELU())
        self.gain_head=nn.Linear(hidden,len(scales)-1)
        self.harm_head=nn.Linear(hidden,len(scales)-1)
        self.harm_logit_head=nn.Linear(hidden,len(scales)-1)
        for h in (self.gain_head,self.harm_head,self.harm_logit_head):
            nn.init.zeros_(h.weight);nn.init.zeros_(h.bias)
        self.register_buffer('calibrated_min_utility',torch.tensor(float('-inf')))
        self.register_buffer('calibrated_max_harm_probability',torch.tensor(1.))
        self.register_buffer('calibration_active',torch.tensor(False))

    @torch.no_grad()
    def set_calibration(self,min_utility=float('-inf'),max_harm_probability=1.,enabled=True):
        self.calibrated_min_utility.fill_(float(min_utility))
        self.calibrated_max_harm_probability.fill_(float(max_harm_probability))
        self.calibration_active.fill_(bool(enabled))

    def forward(self,base,context,transport,mode='fixed',fixed_gate=.75):
        B,P,N=base.shape;proposal=transport['proposed_correction']
        if mode not in ('fixed','learned','none'):raise ValueError(f'Unknown controller mode {mode}')
        hz=self.horizon_embed(torch.arange(P,device=base.device))[None,:,None].expand(B,P,N,8)
        ctx=context[:,None].expand(B,P,N,context.shape[-1])
        aux=torch.stack((base,proposal,proposal.abs(),
             transport['evidence_variance'].clamp_min(0).sqrt(),transport['none_weight'],
             transport['candidate_entropy'],transport['cf_reliability_aggregated'],
             transport['support_compatibility_aggregated'],torch.tanh(base)),-1)
        z=self.net(torch.cat((ctx,hz,aux),-1))
        tail_shape=(*z.shape[:-1],1)
        zero=torch.zeros(tail_shape,device=base.device,dtype=z.dtype)
        positive_gain=torch.cat((zero,F.softplus(self.gain_head(z))),-1)
        positive_harm=torch.cat((zero,F.softplus(self.harm_head(z))),-1)
        harm_logits=torch.cat((torch.full_like(zero,-25.),self.harm_logit_head(z)),-1)
        net=positive_gain-positive_harm
        scores=net-0.10*torch.sigmoid(harm_logits)*positive_harm
        probs=torch.softmax(scores.float(),dim=-1)
        estimated_net=(probs*net).sum(-1)
        estimated_harm_probability=(probs*torch.sigmoid(harm_logits)).sum(-1)
        if mode=='learned':
            gate=(probs*self.scales).sum(-1)
            rejected=(estimated_net<self.calibrated_min_utility)|(
                estimated_harm_probability>self.calibrated_max_harm_probability)
            rejected=rejected & self.calibration_active
            gate=torch.where(rejected,torch.zeros_like(gate),gate)
        else:
            gate=torch.full_like(base,1. if mode=='none' else float(fixed_gate))
            rejected=torch.zeros_like(base,dtype=torch.bool)
        correction=gate*proposal
        return dict(prediction=base+correction,base_prediction=base,correction=correction,
                    gate=gate,ungated_correction=proposal,scale_utility_logits=net,
                    scale_positive_gain=positive_gain,scale_positive_harm=positive_harm,
                    scale_harm_logits=harm_logits,scale_probs=probs,
                    harm_probability=estimated_harm_probability,
                    estimated_gain=(probs*positive_gain).sum(-1),
                    estimated_harm=(probs*positive_harm).sum(-1),
                    estimated_net_utility=estimated_net,
                    reject_correction=rejected)
