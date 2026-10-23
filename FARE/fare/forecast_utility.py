"""FARE v5: utility-aligned retrieval and identity-preserving temporal evidence.

All inference features originate in query history and completed historical episodes.
Future labels appear only in the training-only utility teacher.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


def _evidence_tensor(proposal):
    values=proposal['deltas'].float()                 # [B,N,K,3,H]
    valid=proposal['available'].bool()
    return values, valid


class TemporalEvidenceEncoder(nn.Module):
    """Encode the ordered 12-step sequence; never reduce Evidence to temporal means."""
    def __init__(self,d:int,horizon:int):
        super().__init__()
        self.horizon=horizon
        self.conv=nn.Sequential(nn.Conv1d(6,d,3,padding=1),nn.GELU(),
                                nn.Conv1d(d,d,3,padding=1),nn.GELU())
        self.position=nn.Parameter(torch.randn(horizon,d)*.02)
        self.query=nn.Linear(d,d)
        self.out=nn.Sequential(nn.Linear(d*3+3,d),nn.LayerNorm(d),nn.GELU(),nn.Linear(d,d))
    def forward(self,proposal,query,weights=None):
        values,valid=_evidence_tensor(proposal)
        b,n,k,t,p=values.shape
        assert t==3 and p==self.horizon, 'Expected Residual/Evolution/Failure sequences with configured horizon'
        src=torch.cat([values*valid,valid.float()],dim=3).reshape(b*n*k,6,p)
        hidden=self.conv(src).transpose(1,2).reshape(b,n,k,p,-1)
        hidden=hidden+self.position[None,None,None]
        available=valid.any(3)
        key=self.query(query.mean(2))[:,:,None,None,:]
        logits=(hidden*key).sum(-1)/(hidden.shape[-1]**.5)
        logits=logits.float().masked_fill(~available,-1e6)
        prob=logits.softmax(-1)*available
        prob=prob/prob.sum(-1,keepdim=True).clamp_min(1e-8)
        pooled=(hidden*prob[...,None]).sum(-2)
        # Ordered early/late windows preserve distinct onset/recovery signatures.
        left=(hidden[...,:max(1,p//2),:]*available[...,:max(1,p//2),None]).sum(-2)
        left=left/available[...,:max(1,p//2)].sum(-1,keepdim=True).clamp_min(1)
        right=(hidden[...,p//2:,:]*available[...,p//2:,None]).sum(-2)
        right=right/available[...,p//2:].sum(-1,keepdim=True).clamp_min(1)
        mean_abs=(values.abs()*valid).mean((-1,-2))
        availability=valid.float().mean((-1,-2))
        compat=proposal['compatibility'].float().mean(-1)
        z=self.out(torch.cat([pooled,left,right,mean_abs[...,None],availability[...,None],compat[...,None]],-1))
        if weights is not None:
            # Identity-specific mass controls the actual attention value, not a scalar side feature.
            mass=weights.float().sum((3,4))/p
            z=z*mass[...,None]
        return z*valid.any((-1,-2))[...,None]


class ForecastUtilityRanker(nn.Module):
    """Predict per-horizon counterfactual MAE gain, uncertainty, and harm risk."""
    def __init__(self,d:int,horizon:int):
        super().__init__();self.horizon=horizon
        self.evidence=TemporalEvidenceEncoder(d,horizon)
        self.query=nn.Linear(d,d)
        self.project=nn.Sequential(nn.Linear(d*4+5,2*d),nn.LayerNorm(2*d),
                                   nn.GELU(),nn.Linear(2*d,d),nn.GELU())
        self.gain=nn.Linear(d,horizon)
        self.log_sigma=nn.Linear(d,horizon)
        self.harm=nn.Linear(d,horizon)
        self.gain_scale=nn.Parameter(torch.tensor(.2))
        nn.init.zeros_(self.gain.weight);nn.init.zeros_(self.gain.bias)
        nn.init.constant_(self.log_sigma.bias,-2)
    def forward(self,short,proposal,support):
        e=self.evidence(proposal,support['embedding'])
        b,n,k,d=e.shape
        q=self.query(support['embedding'].mean(2))[:,:,None,:].expand(-1,-1,k,-1)
        available=short['valid'] & proposal['available'].any((-1,-2))
        values=proposal['deltas'].float()
        stats=torch.stack([short['scores'].float().clamp(-10,10),
                           proposal['compatibility'].float().mean(-1),
                           (values[...,0,:]-values[...,1,:]).abs().mean(-1),
                           proposal['available'].float().mean((-1,-2)),
                           support['forecast_deficit'].float().mean(-1)[:,:,None].expand(-1,-1,k)],-1)
        h=self.project(torch.cat([q,e,q-e,(q-e).abs(),stats],-1))
        gain=self.gain(h)*self.gain_scale.abs()
        sigma=F.softplus(self.log_sigma(h))
        harm_logits=self.harm(h)
        # Distinct calibrated outputs: margin penalizes uncertainty and harm, not similarity alone.
        score=gain-.05*sigma-.05*harm_logits.sigmoid()
        return {'gain':gain,'sigma':sigma,'harm_logits':harm_logits,'score':score,
                'valid':available,'candidate_features':h,'candidate_tokens':e}


class ConditionalSetSelector(nn.Module):
    """Differentiable evidence mass with learned NULL, discrete utility-ranked TopK."""
    def __init__(self,d:int,horizon:int,final_k:int,temperature=.25):
        super().__init__();self.final_k=int(final_k);self.horizon=int(horizon)
        self.temperature=float(temperature)
        self.null=nn.Sequential(nn.Linear(d+3,d),nn.GELU(),nn.Linear(d,1))
        nn.init.constant_(self.null[-1].bias,-1.0)
        self.redundancy=nn.Parameter(torch.tensor(.1))
    def forward(self,short,proposal,support,rank,forced_positions=None,force_nonnull=False,global_selection=False):
        values,avail=_evidence_tensor(proposal)
        b,n,k,t,p=values.shape
        available=rank['valid']
        scores=rank['score'].float().masked_fill(~available[...,None],-1e6)
        if forced_positions is not None:
            chosen=forced_positions.long().to(scores.device)
            if chosen.ndim==2:chosen=chosen[...,None]
        else:
            aggregate=scores.mean(-1) if global_selection else scores.max(-1).values
            # Greedy diversity-aware set selection; estimated overlap uses the ordered vectors.
            selected=[];active=torch.zeros_like(available)
            normalized=F.normalize(values.flatten(-2),dim=-1,eps=1e-6)
            for _ in range(min(k,self.final_k)):
                effective=aggregate.masked_fill(active | ~available,-1e6)
                if selected:
                    refs=normalized.gather(2,torch.stack(selected,-1)[...,None].expand(-1,-1,-1,normalized.shape[-1]))
                    similarity=torch.einsum('bnkd,bnsd->bnks',normalized,refs).clamp_min(0)
                    effective=effective-self.redundancy.abs()*similarity.max(-1).values
                pos=effective.argmax(-1)
                selected.append(pos)
                active.scatter_(2,pos[...,None],True)
            chosen=torch.stack(selected,-1)
        j=chosen.shape[-1]
        def gather(z):
            return z.gather(2,chosen[...,None,None].expand(-1,-1,-1,z.shape[-2],z.shape[-1]))
        z=gather(values);good=gather(avail)
        selected_valid=available.gather(2,chosen)
        sc=scores.gather(2,chosen[...,None].expand(-1,-1,-1,p))
        valid_h=good.any(3) & selected_valid[...,None]
        demand=support['demand'].float()
        mean_score=sc.masked_fill(~selected_valid[...,None],0).sum(2)/selected_valid.sum(2).clamp_min(1)[...,None]
        null_feature=torch.cat([support['embedding'].float(),demand[...,None],
                                mean_score[...,None],support['forecast_deficit'][...,None]],-1)
        null_logit=self.null(null_feature).squeeze(-1)
        if force_nonnull:null_logit=torch.full_like(null_logit,-1e7)
        # Softmax over candidate actions and NULL for each horizon; invalid candidates have zero mass.
        prob=torch.cat([sc/self.temperature,null_logit[:,:,None,:]/self.temperature],2)
        prob=prob.masked_fill(~torch.cat([valid_h,
                                          torch.ones(b,n,1,p,device=prob.device,dtype=torch.bool)],2),-1e6)
        mass=prob.softmax(2)
        candidate_mass=mass[:,:,:j,:]
        null_mass=mass[:,:,j,:]
        # Distribute candidate mass only among valid Residual/Evolution/Failure streams.
        branches=good.float();den=branches.sum(3).clamp_min(1)
        w=candidate_mass[:,:,:,None,:]*branches/den[:,:,:,None,:]
        correction=(w*z).sum((2,3))
        return {'selected_positions':chosen,'selected_ids':short['ids'].gather(2,chosen),
                'selected_values':z,'selected_valid':good,'weights':w,'null':null_mass,
                'correction':correction,'predicted_gain':(rank['gain'].gather(2,chosen[...,None].expand(-1,-1,-1,p)) *candidate_mass).sum(2),
                'rank':rank,'selected_mass':candidate_mass,
                'predicted_harm':(rank['harm_logits'].gather(2,chosen[...,None].expand(-1,-1,-1,p)).sigmoid()*candidate_mass).sum(2),
                'predicted_sigma':(rank['sigma'].gather(2,chosen[...,None].expand(-1,-1,-1,p))*candidate_mass).sum(2)}


def candidate_ranking_loss(scores, gains, valid, margin=.02, temperature=.25):
    """Pairwise train-only counterfactual utility supervision for coarse recall.

    `scores`: [B,N,K]; `gains`/`valid`: [B,N,K,H].  Only
    candidates with measured labels are compared; missing is not negative.
    """
    confidence=valid.float().sum(-1)
    expected=(gains.float()*valid.float()).sum(-1)/confidence.clamp_min(1)
    eligible=confidence>0
    delta=expected.unsqueeze(-1)-expected.unsqueeze(-2)
    comparisons=eligible.unsqueeze(-1)&eligible.unsqueeze(-2)&(delta>margin)
    if not comparisons.any():return scores.sum()*0
    differences=scores.float().unsqueeze(-1)-scores.float().unsqueeze(-2)
    return F.softplus(-differences[comparisons]/temperature).mean()


def utility_loss(pred,labels,valid,temperature=.25,margin=.02,return_parts=False):
    """Stable, masked horizon gain/harm/ranking training.

    Heteroscedastic calibration uses FP32 robust Student-t loss with bounded
    sigma; the old 1/(1e-3)^2 amplification is deliberately removed.
    """
    good=valid.bool()&pred['valid'][...,None]
    zero=pred['gain'].float().sum()*0
    if not good.any():
        return (zero,{'regression':zero,'uncertainty':zero,'harm':zero,
                      'listwise':zero,'pairwise':zero}) if return_parts else zero
    gain=pred['gain'].float(); target=labels.float()
    residual=(gain-target).float()
    regression=F.smooth_l1_loss(gain[good],target[good],beta=.1)
    harm=F.binary_cross_entropy_with_logits(pred['harm_logits'].float()[good],(target[good]<-margin).float())
    sigma=pred['sigma'].float().clamp(min=.05,max=10.)
    nu=3.0
    scaled=residual.float()/sigma
    # log1p(x**2 / nu) avoids the unbounded inverse-variance quadratic.
    uncertainty=(torch.log(sigma)+(.5*(nu+1))*torch.log1p(scaled.square()/nu))[good].mean()
    # Stabilized listwise KL: only legal labeled candidates enter softmax.
    useful=good.any(2)
    if useful.any():
        teach_logits=(target/temperature).clamp(-30,30).masked_fill(~good,-1e7)
        score=pred['score'].float().masked_fill(~good,-1e7)
        teach=teach_logits.softmax(2)
        logp=score.log_softmax(2)
        listwise=-(teach*logp).sum(2)[useful].mean()
    else:listwise=zero
    s=pred['score'].float(); d=target.unsqueeze(3)-target.unsqueeze(2)
    eligible=good.unsqueeze(3)&good.unsqueeze(2)&(d>margin)
    if eligible.any():
        difference=s.unsqueeze(3)-s.unsqueeze(2)
        pair=F.softplus(-difference[eligible]/temperature).mean()
    else:pair=zero
    total=regression+.02*uncertainty+.2*harm+.1*listwise+.1*pair
    parts={'regression':regression.detach(),'uncertainty':uncertainty.detach(),
           'harm':harm.detach(),'listwise':listwise.detach(),'pairwise':pair.detach()}
    return (total,parts) if return_parts else total
