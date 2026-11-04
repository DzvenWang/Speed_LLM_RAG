"""Forecast-oriented v5.3 CFRM losses; current-query future used here only for TRAIN labels.
The prediction model accepts no query target and auxiliary losses do not run in evaluation.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

def masked_mean(x,valid):
    return (x.float()*valid.float()).sum()/valid.float().sum().clamp_min(1.)

def cfrm_auxiliary(o,b,mean,std,wrong_pair=False):
    base=o['base_prediction'].detach().float()
    y=b['y_norm'].float()
    valid=b['future_valid'].bool()[:,None] & o['candidate_available'].bool()
    residual=o['candidate_residuals'].float()
    bound=float(o['residual_bound'])
    proposal=bound*torch.tanh(residual/bound)
    gain=((base[:,None]-y[:,None]).abs()-(base[:,None]+proposal-y[:,None]).abs()).detach()
    t=max(0.25,1e-4)
    logits=torch.cat((gain.masked_fill(~valid,-1e4),torch.zeros_like(base)[:,None]),dim=1)/t
    ranking=masked_mean(-(torch.softmax(logits,1)*F.log_softmax(o['candidate_logits'].float(),1)).sum(1),b['future_valid'])
    positive=gain.clamp_min(0);negative=(-gain).clamp_min(0)
    gain_pred=o['candidate_transfer_gain'].float();harm_pred=o['candidate_transfer_harm'].float()
    if valid.any():
        utility=F.smooth_l1_loss(gain_pred[valid],positive[valid])+F.smooth_l1_loss(harm_pred[valid],negative[valid])
    else:utility=ranking*0
    normal=o['normal_component'].float();cf=o['cf_increment'].float()
    normal_pred=bound*torch.tanh(normal/bound)
    cf_pred=bound*torch.tanh((normal+cf)/bound)
    true_util=((base[:,None]+normal_pred-y[:,None]).abs()-(base[:,None]+cf_pred-y[:,None]).abs()).detach()
    if valid.any():
        utility_cf=F.smooth_l1_loss(o['cf_incremental_utility'].float()[valid],true_util[valid])
        bce=F.binary_cross_entropy_with_logits(o['cf_gain_logit'].float()[valid],(true_util[valid]>0).float())
    else:utility_cf=bce=utility*0
    marginal=((-((base[:,None]+o['marginal_reference_proposal'].detach()[:,None]-y[:,None]).abs())+
                 (base[:,None]+o['candidate_loo_proposal'].detach()-y[:,None]).abs())).detach()
    marg=(F.smooth_l1_loss(o['candidate_marginal_utility_pred'].float()[valid],marginal[valid])
          if valid.any() else utility*0)
    wrong=utility*0
    if wrong_pair and valid.any():
        wrong_cf=bound*torch.tanh((normal+o['cf_wrong_increment'].float())/bound)
        wrong_util=((base[:,None]+normal_pred-y[:,None]).abs()-(base[:,None]+wrong_cf-y[:,None]).abs()).detach()
        gap=true_util-wrong_util
        informative=valid&(gap.abs()>.005)
        if informative.any():
            good=o['cf_incremental_utility'].float()[informative]
            bad=o['cf_wrong_utility_pred'].float()[informative]
            wrong=F.softplus((bad-good)*gap[informative].sign()+.05).mean()
    # Original v5.3 relative auxiliary weights; forecast loss in raw units stays primary.
    return std*(.05*utility+.03*ranking+.02*utility_cf+.005*bce+.005*wrong+.01*marg)

