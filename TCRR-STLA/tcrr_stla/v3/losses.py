"""Train-only mechanism supervision. No target-dependent decision at inference."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def masked_mean(x,mask):
    m=mask.to(dtype=x.dtype)
    return (x*m).sum()/m.sum().clamp_min(1.)


def mechanism_losses(out, batch, cf_target=None, *, cf_weight=.02,
                     ranking_weight=.03, harm_weight=.02, control_weight=.01,
                     temperature=.25):
    zero=out['prediction'].new_zeros(())
    level=out.get('v3_contribution','M0')
    valid=batch['future_valid'].bool()
    y=batch['y_norm'].float();base=out['base_prediction'].float().detach()
    cf=zero
    if level!='M0' and cf_target is not None:
        if cf_target.shape!=base.shape:raise ValueError('CF label dimensions disagree')
        cf=masked_mean(F.smooth_l1_loss(out['cf_response'].float(),cf_target.float().detach(),reduction='none'),valid)
    rank=zero
    if level in ('M2','M3'):
        candidate=out['candidate_residuals'].float().detach()
        bound=float(out.get('residual_bound',3.0))
        delta=bound*torch.tanh(candidate/bound)
        gain=(base[:,None]-y[:,None]).abs()-(base[:,None]+delta-y[:,None]).abs()
        valid_c=out['candidate_available'].bool()&valid[:,None]
        scores=out['candidate_transfer_gain'].float()
        harm=out['candidate_transfer_harm'].float()
        if valid_c.any():
            utility=F.smooth_l1_loss(scores[valid_c],gain.detach()[valid_c])
            utility+=.25*F.smooth_l1_loss(harm[valid_c],(-gain).clamp_min(0).detach()[valid_c])
        else:utility=zero
        # Including None (gain=0) yields an explicit query-local competitive label.
        true_logits=torch.cat([gain.masked_fill(~valid_c,-1e4),torch.zeros_like(base)[:,None]],1)/temperature
        target_w=torch.softmax(true_logits.detach(),dim=1)
        pred_logits=out['candidate_logits'].float()
        listwise=masked_mean(-(target_w*torch.log_softmax(pred_logits,dim=1)).sum(1),valid)
        rank=utility+listwise
    tail=zero;control=zero
    if level=='M3':
        proposed=base+out['ungated_correction'].float().detach()
        realized_gain=(base-y).abs()-(proposed-y).abs()
        control=masked_mean(F.smooth_l1_loss(out['estimated_gain'].float(),realized_gain.detach(),reduction='none'),valid)
        control+=masked_mean(F.smooth_l1_loss(out['estimated_harm'].float(),(-realized_gain).clamp_min(0).detach(),reduction='none'),valid)
        corrected=out['prediction'].float()
        harm_actual=((corrected-y).abs()-(base-y).abs()).clamp_min(0)
        tail=masked_mean(harm_actual.square(),valid)
    total=cf_weight*cf+ranking_weight*rank+harm_weight*tail+control_weight*control
    return total,{'cf_recovery':cf.detach(),'transfer_ranking':rank.detach(),
                  'harm_tail':tail.detach(),'gain_harm_supervision':control.detach()}
