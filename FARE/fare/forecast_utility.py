"""Training-only supervision for causal Top-64 forecast utility.

The teacher uses TRAIN labels solely in losses, never in model.forward,
retriever scoring at inference, saved memory keys or checkpoint selection.
Targets compare aligned historical-future analogues to the same BASE forecast
on region-aggregated valid future outcomes. This is a region-level PROXY,
not an oracle for final node-level forecast correction.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F


def supervised_coarse_utility(batch, evidence, regions):
    """Return teacher utility [B,G,K] and eligibility from training labels.

    Regions are disjoint. Historical candidates were causally filtered in the
    retrieval forward. No validation/test examples are added to TrainMemory.
    """
    if 'y_norm' not in batch or 'future_valid' not in batch:
        raise RuntimeError('Coarse utility teachers require training/validation labels')
    ids=evidence.get('coarse_ids')
    candidate=evidence.get('coarse_analog_future')
    cvalid=evidence.get('coarse_future_valid')
    baseline=evidence.get('coarse_base_region')
    if ids is None or candidate is None or cvalid is None or baseline is None:
        return None,None
    with torch.no_grad():
        target=batch['y_norm'].float()  # [B,P,N]
        valid=batch['future_valid'].bool()
        regional=regions.target_mask.to(dtype=target.dtype,device=target.device)
        count=torch.einsum('bpn,gn->bgp',valid.to(target.dtype),regional)
        truth=torch.einsum('bpn,gn->bgp',target*valid,regional)/count.clamp_min(1)
        common=(count[:,:,None,:]>0)&cvalid
        forecast_count=common.sum(-1)
        candidate_err=(((candidate-truth[:,:,None,:]).abs())*common).sum(-1)/forecast_count.clamp_min(1)
        base_err=(((baseline[:,:,None,:]-truth[:,:,None,:]).abs())*common).sum(-1)/forecast_count.clamp_min(1)
        utility=base_err-candidate_err
        eligible=(forecast_count>=max(2,target.shape[1]//4)) & evidence['coarse_valid']
        eligible=eligible & (evidence['coarse_anchor_support']>0)
    return utility,eligible


def coarse_forecast_utility_loss(batch,evidence,regions,temperature=.7):
    """Signed utility regression + pairwise benefit ordering + regret.

    Unlike selected Top-8 soft targets, this supervises ALL legal Top-64
    candidates before Top-8. The signed labels identify all-bad candidate
    sets; uninformative matching should be rejected, not arbitrarily peaked.
    """
    prediction=evidence.get('coarse_utility_pred') if evidence else None
    if prediction is None:
        fallback=evidence['weights'] if evidence else batch['x_obs']
        z=fallback.sum()*0
        return z,{'coarse_utility_eligible_fraction':z.detach()}
    utility,eligibility=supervised_coarse_utility(batch,evidence,regions)
    if utility is None:
        return prediction.sum()*0,{'coarse_utility_eligible_fraction':prediction.new_zeros(())}
    u=utility.detach().clamp(-3.,3.)
    valid=eligibility.float()
    denom=valid.sum().clamp_min(1)
    # Signed regression alone helps identify whether ANY candidate is useful.
    regression=(F.smooth_l1_loss(prediction,u,reduction='none')*valid).sum()/denom
    # Train positive/negative classification on the same valid candidate set.
    positive=(u>0).float()
    positives=(positive*valid).sum()
    negatives=((1-positive)*valid).sum()
    pos_weight=(negatives/positives.clamp_min(1)).clamp(.25,4.)
    ce=F.binary_cross_entropy_with_logits(3.*prediction,positive,reduction='none',pos_weight=pos_weight)
    binary=(ce*valid).sum()/denom
    group_ok=eligibility.sum(-1)>=2
    # Hardest best/worst pair: emphasizes discriminating candidates rather than
    # learning a near-uniform 64-way distribution from tiny score differences.
    best=utility.masked_fill(~eligibility,-1e6).argmax(-1)
    worst=utility.masked_fill(~eligibility,1e6).argmin(-1)
    best_pred=prediction.gather(-1,best[...,None]).squeeze(-1)
    worst_pred=prediction.gather(-1,worst[...,None]).squeeze(-1)
    best_u=utility.gather(-1,best[...,None]).squeeze(-1)
    worst_u=utility.gather(-1,worst[...,None]).squeeze(-1)
    meaningful=group_ok & ((best_u-worst_u)>.03)
    pairwise=(F.softplus(-(best_pred-worst_pred)/max(float(temperature),.05))*meaningful).sum()/meaningful.sum().clamp_min(1)
    # Differentiable rank regret over the LEGAL candidate pool. This rewards
    # expected forecast improvement, not artificial weight concentration.
    masked_pred=prediction.masked_fill(~eligibility,-1e4)
    prob=torch.softmax(masked_pred/.35,dim=-1)*valid
    prob=prob/prob.sum(-1,keepdim=True).clamp_min(1e-9)
    expected=(prob*utility.detach()).sum(-1)
    oracle=best_u
    regret=((oracle-expected).clamp_min(0)*group_ok).sum()/group_ok.sum().clamp_min(1)
    loss=regression + .25*binary + .30*pairwise + .30*regret
    with torch.no_grad():
        best_positive=(best_u>0)&group_ok
        utility_pred_positive=((prediction>0)&eligibility).float().sum()/denom
        selected_utility=(prob.detach()*utility).sum(-1)
        correct=((prediction>0)==(utility>0))&eligibility
        metrics={
            'coarse_utility_eligible_fraction':valid.mean().detach(),
            'coarse_utility_true_positive_fraction':((utility>0)&eligibility).float().sum().div(denom).detach(),
            'coarse_utility_pred_positive_fraction':utility_pred_positive.detach(),
            'coarse_utility_sign_accuracy':correct.float().sum().div(denom).detach(),
            'coarse_utility_oracle_positive_groups':best_positive.float().mean().detach(),
            'coarse_utility_expected_gain':(selected_utility*group_ok).sum().div(group_ok.sum().clamp_min(1)).detach(),
            'coarse_utility_oracle_gain':(oracle*group_ok).sum().div(group_ok.sum().clamp_min(1)).detach(),
            'coarse_utility_regression':regression.detach(),
            'coarse_utility_pairwise':pairwise.detach(),
            'coarse_utility_regret':regret.detach(),
        }
    return loss,metrics
