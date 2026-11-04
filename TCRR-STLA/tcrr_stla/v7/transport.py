"""C1: horizon-aware support compatibility and two-stage mixture-aware transfer.

Stage 1 learns content/CF/support-conditioned utility and makes an initial mixture.
Stage 2 predicts leave-one-out marginal contribution conditioned on that mixture
and refines candidate weights once (not a circular unrolled optimisation).
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class SupportConditionedTransport(nn.Module):
    def __init__(self,horizon,latent,support_dim,hidden=64,bound=3.,utility_weight=.35,
                 marginal_rank_weight=.15,prediction_temperature=1.0):
        super().__init__()
        self.bound=float(bound);self.utility_weight=float(utility_weight)
        self.marginal_rank_weight=float(marginal_rank_weight)
        self.prediction_temperature=float(prediction_temperature)
        if not (0 < self.prediction_temperature <= 10):
            raise ValueError("candidate_prediction_temperature must be in (0,10]")
        self.horizon_embed=nn.Embedding(horizon,8)
        content_dim=latent+8+6
        self.content=nn.Sequential(nn.Linear(content_dim,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        self.utility=nn.Linear(hidden,2)
        self.base_logit=nn.Linear(hidden,1)
        self.cf_logit=nn.Sequential(nn.Linear(hidden+4,hidden//2),nn.GELU(),nn.Linear(hidden//2,1))
        self.support_net=nn.Sequential(nn.Linear(support_dim+latent,hidden),nn.GELU(),
                                       nn.Linear(hidden,hidden),nn.GELU())
        # Unlike v5 initial release, ALL support outputs can vary by horizon.
        self.support_horizon=nn.Linear(8,hidden,bias=False)
        self.support_logit=nn.Linear(hidden,1)
        self.support_scale=nn.Linear(hidden,1)
        self.support_utility=nn.Linear(hidden,2)
        self.interaction=nn.Linear(hidden,1)
        self.none=nn.Sequential(nn.Linear(latent+8,hidden//2),nn.GELU(),nn.Linear(hidden//2,1))
        self.marginal_head=nn.Sequential(nn.Linear(hidden+9,hidden//2),nn.GELU(),nn.Linear(hidden//2,1))
        for head in (self.base_logit,self.cf_logit[-1],self.support_logit,self.support_scale,
                     self.support_utility,self.interaction,self.marginal_head[-1]):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        nn.init.zeros_(self.none[-1].weight);nn.init.constant_(self.none[-1].bias,-1.)

    def _bound(self,m):
        return self.bound*torch.tanh(m/self.bound)

    def _mix(self,weights,correction):
        k=correction.shape[1]
        mix=(weights[:,:k]*correction).sum(1)
        variance=(weights[:,:k]*(correction-mix[:,None]).square()).sum(1)
        variance=variance+weights[:,-1]*mix.square()
        return mix,self._bound(mix),variance

    def _loo(self,weights,correction,mix):
        k=correction.shape[1]
        remain=1-weights[:,:k]
        leave=(mix[:,None]-weights[:,:k]*correction)/remain.clamp_min(1e-6)
        leave=torch.where(remain>1e-6,leave,torch.zeros_like(leave))
        return self._bound(leave)

    def forward(self,base,context,normal,cf_increment,cf_reliability,support,
                score,coverage,valid,cf_enabled=True,support_enabled=True,
                cf_utility=None):
        B,K,P,N=normal.shape
        available=valid.bool() if valid is not None else torch.ones_like(normal,dtype=torch.bool)
        # Invalid or nonfinite cached entries must not enter neural encoders.
        normal=torch.where(available,torch.nan_to_num(normal.float()),torch.zeros_like(normal.float()))
        cf_increment=torch.where(available,torch.nan_to_num(cf_increment.float()),torch.zeros_like(normal))
        cf_reliability=torch.where(available,torch.nan_to_num(cf_reliability.float()).clamp(0,1),torch.zeros_like(normal))
        hz=self.horizon_embed(torch.arange(P,device=base.device))
        horizon=hz[None,None,:,None].expand(B,K,P,N,8)
        ctx4=context[:,None,None].expand(B,K,P,N,context.shape[-1])
        normal=normal.float();correction=normal+(cf_increment.float() if cf_enabled else 0)
        content_input=torch.stack((base[:,None].expand_as(normal),normal,normal.abs(),
                    score[:,:,None,None].expand_as(normal),coverage[:,:,None,None].expand_as(normal),
                    correction),-1)
        content=self.content(torch.cat((content_input,ctx4,horizon),-1).float())
        basic_utility=self.utility(content)
        utility_raw=basic_utility
        support_logit=torch.zeros_like(correction)
        support_strength=torch.ones_like(correction)
        support_utility_delta=torch.zeros_like(basic_utility)
        cf_interaction=torch.zeros_like(correction)
        if support_enabled:
            sup=context[:,None].expand(B,K,N,context.shape[-1])
            sup=self.support_net(torch.cat((support['embedding'],sup),-1))
            # The horizon embedding gives a distinct support response at every P.
            sup_h=torch.tanh(sup[:,:,None]+self.support_horizon(hz)[None,None,:,None])
            support_logit=self.support_logit(sup_h).squeeze(-1)
            support_strength=1+.5*torch.tanh(self.support_scale(sup_h)).squeeze(-1)
            support_utility_delta=self.support_utility(sup_h)
            if cf_enabled:
                cf_interaction=self.interaction(sup_h).squeeze(-1)*torch.tanh(cf_increment)
            utility_raw=utility_raw+support_utility_delta
        gain=F.softplus(utility_raw[...,0]);harm=F.softplus(utility_raw[...,1])
        logits=score[:,:,None,None].float()+self.base_logit(content).squeeze(-1)
        logits=logits+self.utility_weight*(gain-harm)
        if cf_enabled:
            cf_utility=torch.zeros_like(cf_increment) if cf_utility is None else cf_utility
            cf_features=torch.cat((content,cf_increment.unsqueeze(-1),
                cf_reliability.unsqueeze(-1),cf_increment.abs().unsqueeze(-1),cf_utility.unsqueeze(-1)),-1)
            logits=logits+self.cf_logit(cf_features).squeeze(-1)+.1*torch.tanh(cf_utility)
        if support_enabled:logits=logits+support_logit+cf_interaction
        transmitted=correction*support_strength
        safe=torch.where(available,torch.nan_to_num(transmitted),torch.zeros_like(transmitted))
        masked_logits=logits.masked_fill(~available,-1e4)
        none_in=torch.cat((context[:,None].expand(B,P,N,context.shape[-1]),
                         hz[None,:,None].expand(B,P,N,8)),-1)
        none_logit=self.none(none_in).squeeze(-1)
        initial_logits=torch.cat((masked_logits,none_logit[:,None]),1)
        initial_weights=torch.softmax(initial_logits.float()/self.prediction_temperature,1)
        initial_mix,initial_proposal,_=self._mix(initial_weights,safe)
        initial_loo=self._loo(initial_weights,safe,initial_mix)
        # Set-conditioned statistics: no [B,K,K,P,N] pairwise materialization.
        candidate_w=initial_weights[:,:K]
        candidate_mass=candidate_w.sum(1).clamp_min(1e-6)
        set_mean=(candidate_w*safe).sum(1)/candidate_mass
        set_var=(candidate_w*(safe-set_mean[:,None]).square()).sum(1)/candidate_mass
        set_disagreement=torch.sqrt(set_var.clamp_min(0)+1e-8)
        set_agreement=((candidate_w*torch.sign(safe)).sum(1).abs()/candidate_mass).clamp(0,1)
        marginal_context=torch.stack((safe-initial_mix[:,None],safe.abs(),
            initial_mix[:,None].expand_as(safe),candidate_w,
            initial_weights[:,-1,None].expand_as(safe),
            (safe-set_mean[:,None]).abs(),
            set_disagreement[:,None].expand_as(safe),
            set_agreement[:,None].expand_as(safe),
            torch.tanh(safe*set_mean[:,None])), -1)
        marginal=self.marginal_head(torch.cat((content,marginal_context),-1)).squeeze(-1)
        refined_logits=torch.cat((masked_logits+self.marginal_rank_weight*marginal,
                                  none_logit[:,None]),1)
        weights=torch.softmax(refined_logits.float()/self.prediction_temperature,1)
        mixture,proposal,variance=self._mix(weights,safe)
        # CFRM/SCRT -> CRSI: preserve the two transferred components under the
        # SAME final candidate weights and support scaling, including the None
        # candidate. Attribution of the nonlinear bound is symmetric and exact.
        transmitted_normal=torch.where(available,normal*support_strength,torch.zeros_like(normal))
        transmitted_cf=(torch.where(available,cf_increment.float()*support_strength,
                                    torch.zeros_like(normal)) if cf_enabled else torch.zeros_like(normal))
        raw_normal=(weights[:,:K]*transmitted_normal).sum(1)
        raw_cf=(weights[:,:K]*transmitted_cf).sum(1)
        bound_normal=self._bound(raw_normal)
        bound_cf=self._bound(raw_cf)
        # Symmetric nonlinear allocation (two-input Shapley value).
        proposal_normal=.5*(bound_normal+proposal-bound_cf)
        proposal_cf=.5*(bound_cf+proposal-bound_normal)
        # The pair adds to the EXACT bounded full proposal, including when
        # Normal and CF have opposite signs and tanh is strongly nonlinear.
        weighted_cf=weights[:,:K]*transmitted_cf*cf_reliability.float().clamp(0,1)
        weighted_normal=weights[:,:K]*transmitted_normal
        normal_total=weighted_normal.abs().sum(1)
        cf_total=weighted_cf.abs().sum(1)
        normal_agreement=torch.where(normal_total>1e-6,
            weighted_normal.sum(1).abs()/normal_total.clamp_min(1e-6),
            torch.zeros_like(normal_total)).clamp(0,1)
        cf_agreement=torch.where(cf_total>1e-6,
            weighted_cf.sum(1).abs()/cf_total.clamp_min(1e-6),
            torch.zeros_like(cf_total)).clamp(0,1)
        effective_count=(weights[:,:K].sum(1).square()/
                        weights[:,:K].square().sum(1).clamp_min(1e-6))
        cross_conflict=((raw_normal*raw_cf)<-1e-6).float()
        probs=weights.clamp_min(1e-8)
        entropy=-(probs*probs.log()).sum(1)
        cf_rel=(weights[:,:K]*cf_reliability).sum(1) if cf_enabled else torch.zeros_like(base)
        sup_rel=((weights[:,:K]*support['compatibility_base'][:,:,None]).sum(1)
                 if support_enabled else torch.zeros_like(base))
        return dict(candidate_residuals=safe,candidate_available=available,
                    candidate_weights=weights,candidate_logits=refined_logits,
                    candidate_transfer_gain=gain,candidate_transfer_harm=harm,
                    candidate_expected_net_gain=gain-harm,
                    candidate_marginal_utility_pred=marginal,
                    set_disagreement=set_disagreement,set_direction_agreement=set_agreement,
                    candidate_loo_proposal=initial_loo,
                    marginal_reference_proposal=initial_proposal,
                    initial_candidate_logits=initial_logits,
                    initial_candidate_weights=initial_weights,
                    initial_proposal=initial_proposal,
                    proposed_correction=proposal,support_variance=variance,
                    proposed_normal=proposal_normal,proposed_cf=proposal_cf,
                    raw_normal=raw_normal,raw_cf=raw_cf,
                    normal_agreement=normal_agreement,cf_agreement=cf_agreement,
                    normal_signal=normal_total,cf_signal=cf_total,
                    effective_candidate_count=effective_count,
                    component_conflict=cross_conflict,
                    candidate_entropy=entropy,none_weight=weights[:,-1],
                    cf_reliability_aggregated=cf_rel,support_compatibility_aggregated=sup_rel,
                    support_logits_delta=support_logit,support_utility_delta=support_utility_delta,
                    support_strength=support_strength,transported_correction=safe,
                    cf_response=(weights[:,:K]*cf_increment).sum(1) if cf_enabled else torch.zeros_like(base))
