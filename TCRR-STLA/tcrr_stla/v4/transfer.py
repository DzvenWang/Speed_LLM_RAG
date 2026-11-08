"""TCRR v4.0 full-grid counterfactual residual transport.

Only observed query history and train-only paired replay can enter inference.
The final forecast correction is strictly anchored to retrieved historical error.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from ..v3.transfer import support_features

class PairedTransport(nn.Module):
    """C1/C2: query-conditioned paired residual transfer with gain-aware ranking."""
    def __init__(self, horizon: int, latent: int, hidden: int=64, bound: float=3.0,
                 utility_temperature: float=1.0, utility_weight: float=0.35):
        super().__init__()
        self.horizon=int(horizon); self.bound=float(bound)
        self.utility_temperature=float(utility_temperature);self.utility_weight=float(utility_weight)
        self.horizon_embed=nn.Embedding(horizon,8)
        # base, normal, failure, product, disagreement, absolute values,
        # observed history agreement, support, similarity, coverage and query latent
        self.num_scalar=19
        dim=self.num_scalar+latent+8
        self.pair=nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        self.component=nn.Linear(hidden,2)
        self.rank=nn.Linear(hidden,1)
        self.gain_harm=nn.Linear(hidden,3)
        self.none=nn.Sequential(nn.Linear(latent+8+4,hidden//2),nn.GELU(),nn.Linear(hidden//2,1))
        nn.init.zeros_(self.component.weight);nn.init.zeros_(self.component.bias)
        nn.init.zeros_(self.rank.weight);nn.init.zeros_(self.rank.bias)
        nn.init.zeros_(self.gain_harm.weight);nn.init.zeros_(self.gain_harm.bias)
        nn.init.zeros_(self.none[-1].weight);nn.init.constant_(self.none[-1].bias,-1.0)

    def _encode(self,x,mask,base,latent,normal,failure,score,coverage,
                candidate_histories,neighbours,source_valid=None,no_support=False,no_pair=False):
        B,K,P,N=normal.shape; H=mask.shape[1]
        support=support_features(x,mask,neighbours)
        if source_valid is None:
            # Formal v4 always supplies original valid support; fallback is
            # retained solely for legacy synthetic test compatibility.
            source_valid=torch.ones((B,K,H,N),device=x.device,dtype=mask.dtype)
        src=source_valid.float().mean(2)  # [B,K,N]
        qs=support[:,None,None].expand(B,K,P,N,4)
        sc=score[:,:,None,None].expand(B,K,P,N)
        cv=coverage[:,:,None,None].expand(B,K,P,N)
        ms=src[:,:,None].expand(B,K,P,N)
        if candidate_histories is not None:
            agreement=(torch.abs(candidate_histories.float()-x[:,None].float())*mask[:,None]).sum(2)/mask[:,None].sum(2).clamp_min(1.0)
            agreement=torch.exp(-agreement.clamp_min(0).clamp_max(15))[:,:,None].expand(B,K,P,N)
        else:agreement=torch.zeros_like(ms)
        if no_support:
            qs=torch.zeros_like(qs);ms=torch.zeros_like(ms);agreement=torch.zeros_like(agreement)
            support=torch.zeros_like(support)
        norm=normal.float();fail=failure.float();bb=base[:,None].expand(B,K,P,N)
        nf=norm.clamp(-20,20);ff=fail.clamp(-20,20)
        zero=torch.zeros_like(norm)
        t=torch.arange(P,device=x.device).float()/max(P-1,1)
        time=t[None,None,:,None].expand_as(norm)
        scalar=[bb,nf,ff,nf*ff,torch.abs(nf-ff),torch.abs(nf),torch.abs(ff),
                sc,cv,agreement,ms,qs[...,0],qs[...,1],qs[...,2],qs[...,3],time,
                (nf+ff).sign(),torch.tanh(bb),torch.tanh(nf+ff)]
        if no_pair:
            scalar[3]=zero   # normal x failure
            scalar[4]=zero   # paired disagreement
            scalar[16]=zero  # joint response sign
        features=torch.stack(scalar,-1)
        hz=self.horizon_embed(torch.arange(P,device=x.device))[None,None,:,None,:].expand(B,K,P,N,8)
        lat=latent[:,None,None].expand(B,K,P,N,latent.shape[-1])
        z=self.pair(torch.cat([features,lat,hz],-1).float())
        return z,support,hz,qs

    def hard_negative_gain(self,x,mask,base,latent,normal,failure,score,coverage,
                           candidate_histories=None,neighbours=None,source_valid=None):
        """Train on first candidate using a same-query wrong CF pairing, O(BPHN)."""
        if normal.shape[1]<2:return None
        bad_failure=failure.roll(1,1)[:,:1]
        src=source_valid[:,:1] if source_valid is not None else None
        hist=candidate_histories[:,:1] if candidate_histories is not None else None
        feat,_,_,_=self._encode(x,mask,base,latent,normal[:,:1],bad_failure,
                         score[:,:1],coverage[:,:1],hist,neighbours,src)
        return self.gain_harm(feat)[...,0]

    def forward(self,x,mask,base,latent,normal,failure,score,coverage,valid,
                candidate_histories=None,neighbours=None,source_valid=None,mode='full'):
        B,K,P,N=normal.shape
        if failure.shape!=normal.shape or base.shape!=(B,P,N):raise ValueError('C1 paired input shape mismatch')
        if mode=='shuffled_cf_pairing' and K>1:failure=failure.roll(1,1)
        if mode in ('no_failure_residual','no_mask_replay'):failure=torch.zeros_like(failure)
        if mode=='no_normal_residual':normal=torch.zeros_like(normal)
        z,support,hz,qs=self._encode(x,mask,base,latent,normal,failure,score,coverage,candidate_histories,neighbours,source_valid,
                                    no_support=(mode=='no_support_conditioning'),
                                    no_pair=(mode=='no_pair_interaction'))
        scale=2*torch.sigmoid(self.component(z))
        correction=scale[...,0]*normal.float()+scale[...,1]*failure.float()
        if mode=='no_pair_interaction':
            # Equal-capacity no-pair baseline uses a detached failure component
            # for pairing probes, but preserves the same normal/failure proposal.
            correction=normal.float()+failure.float()
        trans=self.gain_harm(z)
        gain=trans[...,0];harm=F.softplus(trans[...,1]);unc=F.softplus(trans[...,2])
        logits=score[:,:,None,None]+self.rank(z).squeeze(-1)
        if mode not in ('no_utility_ranking','similarity_only','no_candidate_utility'):
            logits=logits+self.utility_weight*(gain-harm)/max(self.utility_temperature,1e-4)
        if mode=='similarity_only':logits=score[:,:,None,None].expand_as(logits)
        available=valid.bool() if valid is not None else torch.ones_like(correction,dtype=torch.bool)
        if mode=='no_evidence':available=torch.zeros_like(available)
        logits=logits.masked_fill(~available,-1e4)
        ctx=latent[:,None].expand(B,P,N,latent.shape[-1])
        hh=self.horizon_embed(torch.arange(P,device=x.device))[None,:,None].expand(B,P,N,8)
        none_feat=torch.cat([ctx,hh,support[:,None].expand(B,P,N,4)],-1)
        none=self.none(none_feat.float()).squeeze(-1)
        weights=torch.softmax(torch.cat([logits.float(),none[:,None]],dim=1),dim=1)
        valid_correction=torch.where(available,torch.nan_to_num(correction),torch.zeros_like(correction))
        mixture=(weights[:,:K]*valid_correction).sum(1)
        variance=(weights[:,:K]*(valid_correction-mixture[:,None]).square()).sum(1)
        proposal=self.bound*torch.tanh(mixture/self.bound)
        return dict(candidate_residuals=valid_correction,candidate_available=available,
                    candidate_weights=weights,candidate_logits=torch.cat([logits,none[:,None]],1),
                    candidate_transfer_gain=gain,candidate_transfer_harm=harm,
                    candidate_transfer_uncertainty=unc,normal_residual=normal.float(),
                    failure_residual=failure.float(),cf_response=(weights[:,:K]*failure.float()).sum(1),
                    proposed_correction=proposal,evidence_variance=variance,
                    support_context=support,pair_features=z,
                    none_weight=weights[:,-1])

class SelectiveGainHarm(nn.Module):
    """C3: predict benefit/harm magnitude and harmfulness, with no-label inference."""
    def __init__(self,latent:int,hidden:int=64,bound:float=3.0):
        super().__init__(); self.bound=float(bound)
        self.horizon_embed=nn.Embedding(512,8)
        self.head=nn.Sequential(nn.Linear(latent+8+4+6,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,4))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.reference=nn.Sequential(nn.Linear(latent+8+4+6,hidden//2),nn.GELU(),nn.Linear(hidden//2,1))
        nn.init.zeros_(self.reference[-1].weight);nn.init.constant_(self.reference[-1].bias,1.1)
        self.gain_scale=nn.Parameter(torch.tensor(0.2))
        self.harm_scale=nn.Parameter(torch.tensor(0.5))
    def forward(self,base,context,transport,stage='joint',mode='full',reject_threshold=None):
        B,P,N=base.shape
        hz=self.horizon_embed(torch.arange(P,device=base.device))[None,:,None].expand(B,P,N,8)
        ctx=context[:,None].expand(B,P,N,context.shape[-1]);sup=transport['support_context'][:,None].expand(B,P,N,4)
        proposal=transport['proposed_correction']
        feat=torch.cat([ctx,hz,sup,torch.stack([base,proposal,proposal.abs(),
            transport['evidence_variance'].clamp_min(0).sqrt(),transport['none_weight'],torch.tanh(base)],-1)],-1).float()
        pred=self.head(feat)
        gain=F.softplus(pred[...,0]);harm=F.softplus(pred[...,1]);harm_prob=torch.sigmoid(pred[...,2]);unc=F.softplus(pred[...,3])
        reference=self.reference(feat).squeeze(-1)
        logit=reference+F.softplus(self.gain_scale)*gain-F.softplus(self.harm_scale)*harm-0.1*unc
        if stage=='transfer' or mode=='no_gain_harm':logit=reference
        gate=torch.sigmoid(logit)
        if mode in ('no_risk','always_correct'):gate=torch.ones_like(gate)
        # Thresholds, when enabled, must come from validation, never from test labels.
        reject=torch.zeros_like(gate,dtype=torch.bool)
        if reject_threshold is not None and mode!='no_risk':
            reject=harm_prob>float(reject_threshold)
            gate=torch.where(reject,torch.zeros_like(gate),gate)
        correction=gate*proposal
        return dict(prediction=base+correction,base_prediction=base,correction=correction,gate=gate,
                    risk_logit=logit,risk_probability=harm_prob,harm_probability=harm_prob,
                    harm_logit=pred[...,2],
                    estimated_gain=gain,estimated_harm=harm,estimated_uncertainty=unc,
                    reject_correction=reject,ungated_correction=proposal,reference_risk_logit=reference)

class CounterfactualResidualTransferV4(nn.Module):
    def __init__(self,horizon:int,latent:int,hidden:int=64,bound:float=3.0,
                 utility_weight:float=0.35,utility_temperature:float=1.0):
        super().__init__()
        self.c12=PairedTransport(horizon,latent,hidden,bound,utility_temperature,utility_weight)
        self.c3=SelectiveGainHarm(latent,hidden,bound)
        self.bound=bound;self.stage='transfer';self.reject_threshold=None
    def set_stage(self,stage):
        if stage not in ('transfer','controller','joint'):raise ValueError(stage)
        self.stage=stage
        for p in self.c12.parameters():p.requires_grad_(stage!='controller')
        for p in self.c3.parameters():p.requires_grad_(stage!='transfer')
    def forward(self,x,mask,base,latent,normal,masked_residual,candidate_y,score,coverage,valid,
                candidate_histories=None,neighbours=None,source_valid=None,mode='full'):
        if mode=='no_retrieval':
            z=torch.zeros_like(base);return dict(prediction=base,base_prediction=base,correction=z,gate=z,
                candidate_weights=None,candidate_logits=None,candidate_available=None)
        failure=masked_residual.float()-normal.float()
        out=self.c12(x,mask,base.float(),latent.float(),normal,failure,score.float(),coverage.float(),valid,
                     candidate_histories,neighbours,source_valid,mode)
        gate=self.c3(base.float(),latent.float(),out,stage=self.stage,mode=mode,
                     reject_threshold=self.reject_threshold)
        out.update(gate)
        out.update(v3_contribution='M3',residual_bound=self.bound)
        return out
