"""TRAIN-only multi-mask counterfactual error comparisons.

Two views of *one* source history and *one* future Y are compared.  The
signed error difference is mathematically just a prediction difference;
therefore the substantive additional signal is the ABSOLUTE error degradation
and the relative error ranking.  Inference does not access any target here.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from corsta_v2.util import masked_mean


def teacher_gain(p_masked, p_full, y):
    """Positive = recovering full-view forecast reduces absolute error."""
    return (p_masked.float()-y.float()).abs()-(p_full.float()-y.float()).abs()


def candidate_gain(p_base, p_candidate, y):
    """TRAIN-only utility for choosing an ACTUAL student correction."""
    return (p_base.detach().float()-y.float()).abs()-(p_candidate.detach().float()-y.float()).abs()


def error_contrast_losses(primary,secondary,batch,targets,cfg):
    """Differentiable losses linking both masked views to measured error impact.

    primary/secondary are CoRSTAv2 forward dicts. Teacher predictions use
    identical fixed weights and the shared complete-view reference. Validity
    masking is applied before reduction; only TRAIN may call this function.
    """
    train=cfg['training'];valid=batch['future_valid'].bool()
    y=torch.where(valid,batch['y_norm'].float(),torch.zeros_like(batch['y_norm'].float()))
    p_full=targets['p_full'].detach().float()
    u_a=teacher_gain(primary['backbone'],p_full,y).detach()
    u_b=teacher_gain(secondary['backbone'],p_full,y).detach()
    # Same-world contrast cancels full-view reference: u_a-u_b = |P_a-Y|-|P_b-Y|.
    u_gap=(u_a-u_b).detach()
    temperature=float(train.get('cf_error_temperature',0.35))
    if temperature<=0:raise ValueError('cf_error_temperature must be positive')
    pred_a=primary['cf_teacher_gain'].float()
    pred_b=secondary['cf_teacher_gain'].float()
    gain_loss=masked_mean(F.smooth_l1_loss(pred_a,u_a,reduction='none'),valid)
    gain_loss+=masked_mean(F.smooth_l1_loss(pred_b,u_b,reduction='none'),valid)
    gain_loss*=0.5
    # Soft-ranking targets represent error differences, not mask severity.
    rank_target=torch.sigmoid((u_gap/temperature).clamp(-20,20))
    rank_logit=((pred_a-pred_b)/temperature).clamp(-40,40)
    rank_loss=masked_mean(F.binary_cross_entropy_with_logits(
        rank_logit,rank_target,reduction='none'),valid)
    # Same full-view target; secondary student correction must also fit CF response.
    response_b=(p_full-secondary['backbone'].detach().float()).detach()
    response_loss=masked_mean(F.smooth_l1_loss(
        secondary['delta_cf_candidate'].float(),response_b,reduction='none'),valid)
    # Action-gating targets compare the *current model candidate* with the base,
    # not the ideal teacher full-view forecast. This distinction is essential.
    util_a=candidate_gain(primary['backbone'],primary['cf_raw_prediction'],y)
    util_b=candidate_gain(secondary['backbone'],secondary['cf_raw_prediction'],y)
    soft_a=torch.sigmoid((util_a/temperature).clamp(-20,20))
    soft_b=torch.sigmoid((util_b/temperature).clamp(-20,20))
    acceptance_loss=(masked_mean(F.binary_cross_entropy_with_logits(
        primary['cf_accept_logit'].float(),soft_a,reduction='none'),valid)
         +masked_mean(F.binary_cross_entropy_with_logits(
        secondary['cf_accept_logit'].float(),soft_b,reduction='none'),valid))*0.5
    loss=(float(train.get('cf_error_gain_weight',0.03))*gain_loss
          +float(train.get('cf_error_rank_weight',0.02))*rank_loss
          +float(train.get('cf_contrast_response_weight',0.025))*response_loss
          +float(train.get('cf_acceptance_weight',0.02))*acceptance_loss)
    metrics={'contrast_gain_loss':gain_loss.detach(),
             'contrast_rank_loss':rank_loss.detach(),
             'contrast_cf_response':response_loss.detach(),
             'contrast_acceptance':acceptance_loss.detach(),
             'contrast_true_gap_abs':masked_mean(u_gap.abs(),valid).detach(),
             'contrast_mask_a_gain':masked_mean(u_a,valid).detach(),
             'contrast_mask_b_gain':masked_mean(u_b,valid).detach()}
    return loss,metrics
