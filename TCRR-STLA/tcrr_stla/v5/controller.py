"""C3 — Counterfactual Response-guided Selective Intervention (CRSI).

Controls the SCRT Normal/CF components separately, with directional response
consistency and functional-neighborhood/Horizon regularization. All correction
components are computed at identical, frozen-at-this-step SCRT weights. No
traffic-system intervention or physical-causality claim is implied.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .scale_reference import CorrectionScaleController as ScaleReference
from .regions import functional_region_ids, region_mean

VARIANTS=('scale','component','agreement','localized','full')

class ResponseGuidedController(nn.Module):
    def __init__(self,latent,hidden=64,scales=(0.,.25,.5,.75,1.),
                 default_variant='full',spatial_mix=.5,horizon_group_size=4):
        super().__init__()
        if default_variant not in VARIANTS:raise ValueError(default_variant)
        self.default_variant=default_variant
        self.spatial_mix=float(spatial_mix)
        self.horizon_group_size=int(horizon_group_size)
        if self.horizon_group_size<1:raise ValueError('horizon_group_size must be positive')
        if not 0<=self.spatial_mix<=1:raise ValueError('spatial_mix must be in [0,1]')
        self.reference=ScaleReference(latent,hidden,scales)
        self.register_buffer('scales',torch.tensor(scales,dtype=torch.float32))
        self.horizon_embed=nn.Embedding(512,8)
        self.base_dim=latent+8+9
        self.base_net=nn.Sequential(nn.Linear(self.base_dim,hidden),nn.GELU(),
                                    nn.Linear(hidden,hidden),nn.GELU())
        self.base_logits=nn.Linear(hidden,2)
        # Additions are zero-initialized, isolating each mechanism.
        self.consistency_net=nn.Sequential(nn.Linear(hidden+7,hidden//2),nn.GELU(),nn.Linear(hidden//2,2))
        self.coarse_net=nn.Sequential(nn.Linear(hidden+5,hidden//2),nn.GELU(),nn.Linear(hidden//2,2))
        self.local_refine=nn.Sequential(nn.Linear(hidden+5,hidden//2),nn.GELU(),nn.Linear(hidden//2,2))
        self.component_utility=nn.Linear(hidden,3)  # signed N, CF, N×CF interaction
        self.joint_harm=nn.Sequential(nn.Linear(hidden+3,hidden//2),nn.GELU(),nn.Linear(hidden//2,1))
        self.component_harm=nn.Linear(hidden,2)     # logits for per-component harm
        for head in (self.consistency_net[-1],self.coarse_net[-1],self.local_refine[-1],
                     self.component_utility,self.component_harm,self.joint_harm[-1]):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        nn.init.zeros_(self.base_logits.weight)
        nn.init.constant_(self.base_logits.bias,1.0)
        self._region_cache={}
        self.register_buffer('calibrated_min_utility',torch.tensor(float('-inf')))
        self.register_buffer('calibrated_max_harm_probability',torch.tensor(1.0))
        self.register_buffer('calibration_active',torch.tensor(False))

    @torch.no_grad()
    def set_calibration(self,min_utility=float('-inf'),max_harm_probability=1.,enabled=True):
        self.calibrated_min_utility.fill_(float(min_utility))
        self.calibrated_max_harm_probability.fill_(float(max_harm_probability))
        self.calibration_active.fill_(bool(enabled))

    @staticmethod
    def _temporal_blocks(x,block=4):
        # Input [B,P,N,F]; arbitrary P, including nonmultiples of block.
        B,P,N,D=x.shape
        groups=[]
        for j in range(0,P,block):
            mean=x[:,j:min(j+block,P)].mean(dim=1,keepdim=True)
            groups.append(mean.expand(-1,min(block,P-j),-1,-1))
        return torch.cat(groups,dim=1)

    @staticmethod
    def _neighbor_mean(x,neighbours):
        # Only few channels are smoothed, not full hidden states.
        if neighbours is None or neighbours.numel()==0:return x
        B,P,N,C=x.shape
        if neighbours.ndim!=2 or neighbours.shape[0]!=N:
            raise ValueError('Functional neighbors must be [N,L]')
        idx=neighbours.long().clamp(0,N-1)
        # [B,P,N,L,C] is small because C is only 5.
        gathered=x[:,:,idx,:]
        return (gathered.mean(dim=3)+x)*.5

    def _coarse_fields(self,z,local,neighbours):
        # First aggregate by horizon block, then within disjoint functional
        # neighbour groups. The *coarse* logits contain no raw per-node z,
        # and are therefore exactly shared within each region/time block.
        N=local.shape[2]
        cache_key=(N, None if neighbours is None else neighbours.data_ptr(),
                   None if neighbours is None else neighbours.device)
        if cache_key not in self._region_cache:
            self._region_cache[cache_key]=functional_region_ids(neighbours,N,device=local.device)
        regions=self._region_cache[cache_key].to(local.device)
        coarse_local=region_mean(self._temporal_blocks(local,self.horizon_group_size),regions)
        coarse_z=region_mean(self._temporal_blocks(z,self.horizon_group_size),regions)
        return coarse_z,coarse_local,regions

    def _features(self,base,context,transport):
        B,P,N=base.shape
        h=self.horizon_embed(torch.arange(P,device=base.device))[None,:,None].expand(B,P,N,8)
        ctx=context[:,None].expand(B,P,N,context.shape[-1])
        n=transport['proposed_normal'];cf=transport['proposed_cf']
        mix=transport['proposed_correction']
        ex=torch.stack((base,n,cf,mix,n.abs(),cf.abs(),
            transport['none_weight'],transport['candidate_entropy'],
            transport['support_compatibility_aggregated']),dim=-1)
        z=self.base_net(torch.cat((ctx,h,ex),dim=-1))
        agree=torch.stack((transport['normal_agreement'],transport['cf_agreement'],
            transport['normal_signal'],transport['cf_signal'],
            transport['component_conflict'],transport['effective_candidate_count']/max(transport['candidate_weights'].shape[1]-1,1),
            transport['cf_reliability_aggregated']),-1)
        local=torch.stack((n,cf,transport['cf_agreement'],transport['component_conflict'],
                           transport['support_compatibility_aggregated']),dim=-1)
        return z,agree,local

    def forward(self,base,context,transport,mode='fixed',fixed_gate=.75,
                neighbours=None,variant=None):
        if mode not in ('fixed','learned','none'):
            raise ValueError(f'Unknown controller mode {mode}')
        variant=self.default_variant if variant is None else variant
        if variant not in VARIANTS:raise ValueError(variant)
        n=transport['proposed_normal'];cf=transport['proposed_cf']
        proposal=transport['proposed_correction']
        # C3 OFF must be strictly independent of ALL controller parameters.
        if mode=='none':
            gate=torch.ones_like(base)
            correction=proposal
            reject=torch.zeros_like(base,dtype=torch.bool)
            alpha_n=alpha_cf=gate
        elif mode=='fixed':
            gate=torch.full_like(base,float(fixed_gate))
            correction=gate*proposal
            alpha_n=alpha_cf=gate
            reject=torch.zeros_like(base,dtype=torch.bool)
        else:
            gate=correction=alpha_n=alpha_cf=reject=None
        # Training-time utility targets may still consume predictions from the
        # scale comparator. Its outputs are independent of the C3 OFF decision.
        ref=self.reference(base,context,transport,mode='learned' if mode=='learned' else 'none')
        z,agreement,local=self._features(base,context,transport)
        util=self.component_utility(z)
        harm_logits=self.component_harm(z)
        est_harm=(torch.sigmoid(harm_logits)).mean(-1)
        coarse_logits=torch.zeros_like(harm_logits)
        local_adjust=torch.zeros_like(harm_logits)
        region_ids=None
        if mode=='learned':
            if variant=='scale':
                correction=ref['correction'];gate=ref['gate']
                alpha_n=alpha_cf=gate
                net=ref['estimated_net_utility']
                harm=ref['harm_probability']
            else:
                raw=self.base_logits(z)
                if variant in ('agreement','full'):
                    raw=raw+self.consistency_net(torch.cat((z,agreement),-1))
                if variant in ('localized','full'):
                    # Hierarchical temporal blocks + train-derived functional
                    # neighborhood pooling, with bounded local refinement.
                    coarse_z,coarse,region_ids=self._coarse_fields(z,local,neighbours)
                    coarse_logits=self.coarse_net(torch.cat((coarse_z,coarse),-1))
                    # Limit local deviations; learned coarse part is shared
                    # exactly over the same time block and functional region.
                    local_adjust=.25*torch.tanh(self.local_refine(
                        torch.cat((z,local-coarse),-1)))
                    raw=raw+self.spatial_mix*coarse_logits+local_adjust
                # Forecast-driven component utility also affects activation.
                gates=torch.sigmoid(raw+0.20*torch.tanh(util[...,:2]))
                alpha_n,alpha_cf=gates.unbind(-1)
                correction=alpha_n*n+alpha_cf*cf
                mag=(n.abs()+cf.abs()).clamp_min(1e-6)
                gate=((alpha_n*n.abs()+alpha_cf*cf.abs())/mag).clamp(0,1)
                net=(alpha_n*util[...,0]+alpha_cf*util[...,1]
                     +torch.minimum(alpha_n,alpha_cf)*util[...,2])
                # Predict harm from the *joint* selected correction, not an
                # average of individual-component harms. Detach selected
                # intensities for auxiliary BCE to avoid self-referential labels.
                joint_input=torch.cat((z,alpha_n.detach()[...,None],
                        alpha_cf.detach()[...,None],correction.detach()[...,None]),-1)
                joint_harm_logit=self.joint_harm(joint_input).squeeze(-1)
                harm=torch.sigmoid(joint_harm_logit)
            if variant=='scale':
                joint_harm_logit=torch.logit(harm.clamp(1e-6,1-1e-6))
            pre_rejection_correction=correction
            rejected=(net<self.calibrated_min_utility) | (harm>self.calibrated_max_harm_probability)
            rejected=rejected & self.calibration_active
            correction=torch.where(rejected,torch.zeros_like(correction),correction)
            alpha_n=torch.where(rejected,torch.zeros_like(alpha_n),alpha_n)
            alpha_cf=torch.where(rejected,torch.zeros_like(alpha_cf),alpha_cf)
            gate=torch.where(rejected,torch.zeros_like(gate),gate)
            estimated_net=net
            harm_probability=harm
            reject=rejected
        else:
            # Unused for decision when C3 is disabled/fixed.
            estimated_net=util[...,0]+util[...,1]+util[...,2]
            harm_probability=est_harm
            pre_rejection_correction=correction
            joint_harm_logit=self.joint_harm(torch.cat((z,alpha_n[...,None],
                alpha_cf[...,None],correction[...,None]),-1)).squeeze(-1)
        return dict(prediction=base+correction,base_prediction=base,
                    correction=correction,gate=gate,ungated_correction=proposal,
                    normal_gate=alpha_n,cf_gate=alpha_cf,
                    estimated_net_utility=estimated_net,harm_probability=harm_probability,
                    reject_correction=reject,
                    component_utility_pred=util,component_harm_logits=harm_logits,
                    controller_variant=variant,
                    pre_rejection_correction=pre_rejection_correction,
                    joint_harm_logit=joint_harm_logit,
                    coarse_intervention_logits=coarse_logits,
                    local_intervention_deviation=local_adjust,
                    functional_region_ids=region_ids,
                    # Retained v5 compatibility diagnostics and stage losses.
                    scale_utility_logits=ref['scale_utility_logits'],
                    scale_positive_gain=ref['scale_positive_gain'],
                    scale_positive_harm=ref['scale_positive_harm'],
                    scale_harm_logits=ref['scale_harm_logits'],
                    scale_probs=ref['scale_probs'],
                    estimated_gain=ref['estimated_gain'],estimated_harm=ref['estimated_harm'])
