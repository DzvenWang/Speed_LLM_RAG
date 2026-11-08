"""TCRR v3 C1/C2/C3. All runtime inputs are observed query features or legal Train memory.

Teacher full-input labels are never accepted by forward; those exist only in the
training loss. The counterfactual candidate response is matched-mask error minus
full-observation error from one frozen Teacher, as in the source replay banks.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

LEVELS = {'M0': 0, 'M1': 1, 'M2': 2, 'M3': 3}


def support_features(x: torch.Tensor, mask: torch.Tensor, neighbours=None):
    """Return [B,N,4] using only actually observed history and graph topology."""
    m = mask.float()
    b, h, n = m.shape
    coverage = m.mean(1)
    last = m.flip(1).argmax(1).float() / max(1, h-1)
    last = torch.where(m.bool().any(1), last, torch.ones_like(last))
    recent = m[:, -max(1, h//3):].mean(1)
    if neighbours is not None and neighbours.numel() and neighbours.shape[0] == n:
        local = coverage[:, neighbours.long()].mean(-1)
    else:
        local = coverage.mean(1,keepdim=True).expand_as(coverage)
    return torch.stack([coverage, recent, last, local], -1)


class CounterfactualResidualTransfer(nn.Module):
    """Three contribution switches M0 plain residual; M1 CF; M2 support; M3 risk.

    Keeps an evidence-only correction. No unconstrained future forecasting head.
    The same fixed bounded correction and legacy gate are used by M0/M1/M2.
    """
    def __init__(self, horizon: int, latent: int, hidden: int=64,
                 bound: float=3.0, level: str='M3'):
        super().__init__()
        if level not in LEVELS:raise ValueError(f'Unknown contribution {level}')
        self.level=level;self.bound=float(bound);self.horizon=int(horizon)
        self.horizon_embed=nn.Embedding(horizon,8)
        # each feature is explicitly tied to a retrieval or support observable
        d = latent + 8 + 15
        self.transfer_ranker=nn.Sequential(nn.Linear(d,hidden),nn.GELU(),nn.Linear(hidden,3))
        self.component_scale=nn.Sequential(nn.Linear(d,hidden),nn.GELU(),nn.Linear(hidden,2))
        self.none=nn.Sequential(nn.Linear(latent+8+4,hidden//2),nn.GELU(),nn.Linear(hidden//2,1))
        gate_d = latent+8+4+5
        self.reference_risk=nn.Sequential(nn.Linear(gate_d,hidden//2),nn.GELU(),nn.Linear(hidden//2,1))
        self.gain_harm=nn.Sequential(nn.Linear(gate_d,hidden),nn.GELU(),nn.Linear(hidden,3))
        nn.init.zeros_(self.component_scale[-1].weight)
        nn.init.zeros_(self.component_scale[-1].bias)
        nn.init.zeros_(self.transfer_ranker[-1].weight)
        nn.init.zeros_(self.transfer_ranker[-1].bias)
        nn.init.zeros_(self.gain_harm[-1].weight)
        nn.init.zeros_(self.gain_harm[-1].bias)
        nn.init.zeros_(self.reference_risk[-1].weight)
        nn.init.constant_(self.reference_risk[-1].bias,1.5)
        nn.init.constant_(self.none[-1].bias,0.0)

    def forward(self, x, mask, base, latent, normal, masked_residual,
                candidate_y, score, coverage, valid, candidate_histories=None,
                neighbours=None, mode='full'):
        b,p,n=base.shape; k=normal.shape[1]
        if masked_residual.shape != (b,k,p,n) or candidate_y.shape != (b,k,p,n):
            raise ValueError('TCRR v3 candidate shapes must be [B,K,P,N]')
        support=support_features(x,mask,neighbours)
        hh=self.horizon_embed(torch.arange(p,device=x.device))[None,:,None,:].expand(b,p,n,-1)
        context=latent[:,None].expand(b,p,n,-1)
        failure=(masked_residual.float()-normal.float())
        norm=normal.float()
        if mode in ('shuffled','random_future') and k>1:
            failure=failure.roll(1,dims=1)
            norm=norm.roll(1,dims=1)
        if mode=='shuffled_cf_pairing' and k>1:
            failure=failure.roll(1,dims=1)  # Break correct historical pairing only.
        if mode in ('no_failure_residual','no_mask_replay'):failure=torch.zeros_like(failure)
        if mode=='no_normal_residual':norm=torch.zeros_like(norm)
        diff=torch.zeros(b,k,n,device=x.device)
        trend_similarity=torch.zeros_like(diff)
        neighbour_trend_similarity=torch.zeros_like(diff)
        if candidate_histories is not None:
            denom=mask[:,None].sum(2).clamp_min(1.)
            diff=((candidate_histories.float()-x[:,None].float()).abs()*mask[:,None]).sum(2)/denom
            # Locally observed temporal innovations are compared under the SAME
            # query observation mask, not hidden query values or future labels.
            pair=(mask[:,1:]*mask[:,:-1]).float()
            qtrend=(((x[:,1:]-x[:,:-1])*pair).sum(1)/pair.sum(1).clamp_min(1.)).float()
            ctrend=((candidate_histories[:,:,1:]-candidate_histories[:,:,:-1])*pair[:,None]).sum(2)/pair.sum(1)[:,None].clamp_min(1.)
            trend_similarity=torch.exp(-(ctrend-qtrend[:,None]).abs().clamp_max(15))
            if neighbours is not None and neighbours.numel() and neighbours.shape[0]==n:
                nq=qtrend[:,neighbours.long()].mean(-1)
                nc=ctrend[:,:,neighbours.long()].mean(-1)
                neighbour_trend_similarity=torch.exp(-(nc-nq[:,None]).abs().clamp_max(15))
            else:neighbour_trend_similarity=trend_similarity
        info=torch.stack([
            base[:,None].expand(b,k,p,n).float(),norm,failure,
            norm.abs(),failure.abs(),candidate_y.float()-base[:,None].float(),
            score[:,:,None,None].expand(b,k,p,n).float(),
            coverage[:,:,None,None].expand(b,k,p,n).float(),
            diff[:,:,None].expand(b,k,p,n),
            support[...,0][:,None,None].expand(b,k,p,n),
            support[...,2][:,None,None].expand(b,k,p,n),
            support[...,3][:,None,None].expand(b,k,p,n),
            trend_similarity[:,:,None].expand(b,k,p,n),
            neighbour_trend_similarity[:,:,None].expand(b,k,p,n),
            (torch.arange(p,device=x.device).float()/max(1,p-1))[None,None,:,None].expand(b,k,p,n)
        ],-1)
        if mode=='no_support_conditioning':
            info=info.clone()
            info[...,9:14]=0.0   # support, missing age and lagged-neighbor trend compatibility
        feat=torch.cat([info,context[:,None].expand(b,k,p,n,-1),hh[:,None].expand(b,k,p,n,-1)],-1).float()
        v = valid.bool() if valid is not None else torch.ones((b,k,p,n),device=x.device,dtype=torch.bool)
        if mode == 'no_evidence': v = torch.zeros_like(v)
        if self.level=='M0':
            candidate=norm
            logits=score[:,:,None,None].expand(b,k,p,n).float()
        elif self.level=='M1':
            candidate=norm+failure
            logits=score[:,:,None,None].expand(b,k,p,n).float()
        else:
            sc=2*torch.sigmoid(self.component_scale(feat))
            if mode=='no_failure_residual': sc=sc.clone();sc[...,1]=0
            if mode=='no_normal_residual': sc=sc.clone();sc[...,0]=0
            candidate=sc[...,0]*norm+sc[...,1]*failure
            logits=score[:,:,None,None].expand(b,k,p,n).float()+self.transfer_ranker(feat)[...,0]
            # only C2 learns a support-conditioned evidence ranking
        logits=logits.masked_fill(~v,-1e4)
        none_ctx=torch.cat([context,hh,support[:,None].expand(b,p,n,4)],-1)
        null=self.none(none_ctx.float()).squeeze(-1)
        weights=torch.softmax(torch.cat([logits.float(),null[:,None].float()],1),1)
        w=weights[:,:k]
        residual=torch.where(v,torch.nan_to_num(candidate),torch.zeros_like(candidate))
        evidence=(w*residual).sum(1)
        cf_response=(w*failure).sum(1) if self.level!='M0' else torch.zeros_like(base)
        discrepancy=(w*(residual-evidence[:,None]).square()).sum(1).clamp_min(0)
        gf=torch.cat([context,hh,support[:,None].expand(b,p,n,4),torch.stack([
            base.float(),evidence,evidence.abs(),(discrepancy+1e-6).sqrt(),weights[:,-1]],-1)],-1)
        reference_logit=self.reference_risk(gf.float()).squeeze(-1)
        head=self.gain_harm(gf.float())
        estimated_gain=head[...,0]
        estimated_harm=F.softplus(head[...,1])
        estimated_uncertainty=F.softplus(head[...,2])
        if self.level=='M3' and mode!='no_gain_harm':
            # Current value is an independently learned *risk-averse* adjustment
            # to the common reference risk; no teacher future is used at runtime.
            logit=reference_logit+estimated_gain-estimated_harm-0.1*estimated_uncertainty
        else:
            logit=reference_logit
        gate=torch.sigmoid(logit)
        if mode=='no_risk':gate=torch.ones_like(gate)
        ungated=self.bound*torch.tanh(evidence/self.bound)
        correction=gate*ungated
        return {
            'prediction':base+correction,'base_prediction':base,
            'correction':correction,'gate':gate,'risk_probability':gate,'risk_logit':logit,
            'ungated_correction':ungated,'candidate_weights':weights,
            'candidate_logits':torch.cat([logits,null[:,None]],1),
            'candidate_residuals':residual,'candidate_available':v,
            'normal_residual':norm,'failure_residual':failure,
            'cf_response':cf_response,'estimated_gain':estimated_gain,
            'estimated_harm':estimated_harm,'estimated_uncertainty':estimated_uncertainty,
            'candidate_transfer_gain':self.transfer_ranker(feat)[...,1] if self.level in ('M2','M3') else None,
            'candidate_transfer_harm':F.softplus(self.transfer_ranker(feat)[...,2]) if self.level in ('M2','M3') else None,
            'reference_risk_logit':reference_logit,'residual_bound':self.bound,
            'support_context':support,
            'v3_contribution':self.level,
        }
