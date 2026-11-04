"""Forecast-oriented v5.3 CFRM losses; current-query future used here only for TRAIN labels.
The prediction model accepts no query target and auxiliary losses do not run in evaluation.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

def masked_mean(x, valid):
    """NaN-safe masked mean: invalid labels may contain NaN/Inf."""
    if x.shape != valid.shape:
        raise ValueError(f"Masked mean shape mismatch: {x.shape} vs {valid.shape}")
    mask=valid.bool()
    values=torch.where(mask,x.float(),torch.zeros_like(x.float()))
    if not torch.isfinite(values).all():
        raise RuntimeError("Nonfinite loss on a valid target")
    return values.sum()/mask.float().sum().clamp_min(1.)


def full_activation_targets(o,b):
    """Utility of each *full-activation* candidate against Normal.

    None has a real P0 fallback score, not a synthetic zero utility.  Query
    labels are used only by TRAIN auxiliary losses, never by model.forward.
    """
    y=b['y_norm'].float()
    future_valid=b['future_valid'].bool()
    p0=o['base_prediction'].detach().float()
    pn=o['normal'].detach().float()
    available=o['candidate_available'].bool()
    valid=available & future_valid[:,None]
    bound=float(o['residual_bound'])
    if not (0<bound<1e6):raise ValueError('Invalid residual bound')
    residual=torch.where(available,torch.nan_to_num(o['candidate_residuals'].detach().float()),
                         torch.zeros_like(o['candidate_residuals'].float()))
    # Avoid computing with nonfinite invalid future labels before masking.
    y_clean=torch.where(future_valid,y,torch.zeros_like(y))
    if not torch.isfinite(y_clean).all():raise RuntimeError('Nonfinite valid TRAIN labels')
    base_error=(pn-y_clean).abs()
    pred=p0[:,None]+bound*torch.tanh(residual/bound)
    utility=base_error[:,None]-(pred-y_clean[:,None]).abs()
    none_utility=base_error-(p0-y_clean).abs()
    return (torch.where(valid,utility,torch.zeros_like(utility)).detach(),
            torch.where(future_valid,none_utility,torch.zeros_like(none_utility)).detach(),
            valid)


def ranking_targets(utility,none_utility,valid,target_temperature):
    t=float(target_temperature)
    if not (0<t<=10):raise ValueError('candidate_target_temperature must be in (0,10]')
    values=torch.cat((utility.masked_fill(~valid,-1e4),none_utility[:,None]),dim=1)
    return torch.softmax(values.float()/t,dim=1).detach()


def cfrm_auxiliary(o,b,mean,std,wrong_pair=False,target_temperature=.25,return_parts=False):
    p0=o['base_prediction'].detach().float()
    y=b['y_norm'].float()
    future=b['future_valid'].bool()
    y_safe=torch.where(future,y,torch.zeros_like(y))
    utility_target,none_target,valid=full_activation_targets(o,b)
    targets=ranking_targets(utility_target,none_target,valid,target_temperature)
    ranking=masked_mean(-(targets*F.log_softmax(o['candidate_logits'].float(),dim=1)).sum(1),future)
    positive=utility_target.clamp_min(0)
    negative=(-utility_target).clamp_min(0)
    gain_pred=o['candidate_transfer_gain'].float()
    harm_pred=o['candidate_transfer_harm'].float()
    if valid.any():
        utility=(F.smooth_l1_loss(gain_pred[valid],positive[valid])+
                 F.smooth_l1_loss(harm_pred[valid],negative[valid]))
    else:utility=ranking*0
    bound=float(o['residual_bound'])
    normal=o['normal_component'].float();cf=o['cf_increment'].float()
    normal_pred=bound*torch.tanh(normal/bound)
    cf_pred=bound*torch.tanh((normal+cf)/bound)
    true_util=((p0[:,None]+normal_pred-y_safe[:,None]).abs()-
               (p0[:,None]+cf_pred-y_safe[:,None]).abs()).detach()
    if valid.any():
        utility_cf=F.smooth_l1_loss(o['cf_incremental_utility'].float()[valid],true_util[valid])
        bce=F.binary_cross_entropy_with_logits(o['cf_gain_logit'].float()[valid],(true_util[valid]>0).float())
    else:utility_cf=bce=utility*0
    ref=o['marginal_reference_proposal'].detach()
    loo=o['candidate_loo_proposal'].detach()
    marginal=((p0[:,None]+loo-y_safe[:,None]).abs()-
              (p0[:,None]+ref[:,None]-y_safe[:,None]).abs()).detach()
    marg=(F.smooth_l1_loss(o['candidate_marginal_utility_pred'].float()[valid],marginal[valid])
          if valid.any() else utility*0)
    wrong=utility*0
    if wrong_pair and valid.any():
        wrong_cf=bound*torch.tanh((normal+o['cf_wrong_increment'].float())/bound)
        wrong_util=((p0[:,None]+normal_pred-y_safe[:,None]).abs()-
                    (p0[:,None]+wrong_cf-y_safe[:,None]).abs()).detach()
        gap=true_util-wrong_util
        informative=valid&(gap.abs()>.005)
        if informative.any():
            good=o['cf_incremental_utility'].float()[informative]
            bad=o['cf_wrong_utility_pred'].float()[informative]
            wrong=F.softplus((bad-good)*gap[informative].sign()+.05).mean()
    scale=float(std)
    if not (scale>0):raise ValueError('Target standard deviation must be positive')
    parts={'utility':scale*.05*utility,'ranking':scale*.03*ranking,
           'cf_utility':scale*.02*utility_cf,'cf_bce':scale*.005*bce,
           'wrong_pair':scale*.005*wrong,'marginal':scale*.01*marg}
    total=sum(parts.values())
    if not torch.isfinite(total):raise RuntimeError('Nonfinite C1 auxiliary loss')
    return (total,parts) if return_parts else total
