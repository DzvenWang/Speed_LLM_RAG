"""v0.2.2: TRAIN-only full-to-masked Normal anchor and CF incremental utility.

P0 = f_T(X_masked), Pfull = f_T(X_full) with identical frozen parameters.
RN*=Y-Pfull; DCF*=Pfull-P0.  At inference only masked X is seen.
P_N=P0+Rhat_N, P_candidate=P_N+Dhat_CF.
The utility label measures the *candidate* improvement relative to P_N.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from corsta_v2.util import masked_mean


def _regression(x, target, valid):
    return masked_mean(F.smooth_l1_loss(x.float(),target.float(),reduction='none'),valid)


def _forecast(pred,y,valid,std,rmse_weight):
    err=(pred.float()-y.float())*std
    mae=masked_mean(err.abs(),valid)
    rmse=torch.sqrt(masked_mean(err.square(),valid)+1e-8)
    return mae+float(rmse_weight)*rmse,mae,rmse


def actual_anchor_gain(p_normal,p_candidate,y):
    """Only call during TRAIN or target-free metric calculation in evaluation."""
    return (p_normal.detach().float()-y.float()).abs()-(p_candidate.detach().float()-y.float()).abs()


def _dose_gain_loss(out,y,valid,scale):
    """Dose-policy regression onto the REAL gain over the Normal anchor.

    For every rung d the candidate is P_N + psi_d * Delta_CF (both detached);
    the label is |P_N-Y| - |candidate_d-Y|, measured against the true future.
    TRAIN/selection only; the policy sees inference-time inputs.
    """
    psi=out['dose_psi'].detach().float()
    delta=out['delta_cf_candidate'].detach().float()[:,None]
    normal=out['normal_prediction'].detach().float()[:,None]
    yy=y.float()[:,None]
    actual=(normal-yy).abs()-(normal+psi*delta-yy).abs()
    weights=valid[:,None].expand_as(actual)
    predicted=out['dose_gain_pred'].float()
    loss=masked_mean(F.smooth_l1_loss(predicted/scale,actual/scale,reduction='none'),weights)
    return loss,actual


def dose_response_loss(out,dose_p,valid):
    """Teacher dose ladder: psi_d * Delta_CF should match P_d - P0 for d<1.

    dose_p [B,D',P,N] are TRAIN-only partial-restoration forecasts from the
    identical frozen STLA.  The student grid's first D' rungs are the same
    doses (the top rung d=1 is the existing full-counterfactual response loss).
    """
    n=dose_p.shape[1]
    if out['dose_psi'].shape[1]<n+1:
        raise RuntimeError('student dose grid must contain every teacher rung plus d=1')
    predicted=out['dose_psi'][:,:n].float()*out['delta_cf_candidate'].float()[:,None]
    target=dose_p.detach().float()-out['backbone'].detach().float()[:,None]
    weights=valid[:,None].expand_as(predicted)
    return masked_mean(F.smooth_l1_loss(predicted,target,reduction='none'),weights)


def _anchor_single(out,y,valid,p_full,cfg,stage,std):
    c=cfg['training'];P=out['backbone'].float()
    # Stage N: the teacher-forced forecast uses oracle CF only as TRAIN loss.
    # This is NEVER a deployment output or validation evaluation prediction.
    if stage=='normal':
        reference_delta=(p_full-P).detach()
        proxy=out['normal_prediction']+reference_delta
        forecast,mae,rmse=_forecast(proxy,y,valid,std,c['forecast_rmse_weight'])
        semantic=_regression(out['normal_delta'],y-p_full,valid)
        loss=forecast+float(c.get('normal_residual_weight',0.1))*semantic
        return loss,{'mae':mae.detach(),'rmse':rmse.detach(),
                      'normal_semantic':semantic.detach(),
                      'normal_observed_mae':masked_mean((out['normal_prediction']-y).abs(),valid).detach()}

    forecast,mae,rmse=_forecast(out['prediction'],y,valid,std,c['forecast_rmse_weight'])
    loss=forecast
    details={'mae':mae.detach(),'rmse':rmse.detach()}
    response=(p_full-P).detach()
    delta=out['delta_cf_candidate'].float()
    actual=actual_anchor_gain(out['normal_prediction'],out['normal_cf_candidate'],y).detach()
    predict_gain=out['anchor_gain_pred'].float()
    t=float(c.get('cf_error_temperature',0.35))
    if t<=0:raise ValueError('cf_error_temperature must be positive')
    accept_target=torch.sigmoid((actual/t).clamp(-20,20))
    utility_loss=_regression(predict_gain,actual,valid)
    accept_loss=masked_mean(F.binary_cross_entropy_with_logits(
        out['cf_accept_logit'].float(),accept_target,reduction='none'),valid)
    loss=loss+float(c.get('cf_anchor_gain_weight',0.05))*utility_loss
    loss=loss+float(c.get('cf_acceptance_weight',0.02))*accept_loss
    details.update(anchor_gain=utility_loss.detach(),anchor_acceptance=accept_loss.detach(),
                   anchor_candidate_actual_gain=masked_mean(actual,valid).detach(),
                   anchor_applied_gain=masked_mean((out['normal_prediction'].detach()-y).abs()-
                                   (out['prediction'].detach()-y).abs(),valid).detach())
    if out.get('dose_gain_pred') is not None:
        scale=float(c.get('dose_gain_scale',0.1))
        if scale<=0:raise ValueError('dose_gain_scale must be positive')
        dose_loss,actual_d=_dose_gain_loss(out,y,valid,scale)
        loss=loss+float(c.get('dose_gain_weight',0.05))*dose_loss
        details.update(dose_gain=dose_loss.detach(),
                       dose_full_actual_gain=masked_mean(actual_d[:,-1],valid).detach())
        if out.get('dose_selected_level') is not None:
            details['dose_applied_level']=masked_mean(
                out['dose_selected_level'].detach().float(),valid)
    if stage=='anchor_cf':
        # Candidate response is referenced to P0, not P_N.  P_N is frozen.
        cf_loss=_regression(delta,response,valid)
        teacher_utility=((P.detach()-y).abs()-(p_full.detach()-y).abs()).detach()
        ref_loss=_regression(out['cf_teacher_gain'],teacher_utility,valid)
        loss+=float(c.get('cf_response_weight',0.06))*cf_loss
        loss+=float(c.get('cf_error_gain_weight',0.03))*ref_loss
        details.update(cf_response=cf_loss.detach(),teacher_gain=ref_loss.detach())
    return loss,details


def normal_anchor_loss(out,b,targets,stage,mean,std,cfg,other_views=None):
    """K views of the SAME (Xfull,Y), with one stationary Normal anchor model.

    `targets` is computed once on the primary view by the TRAIN PairedTeacher.
    Additional teacher masked predictions are supplied by each fixed STLA P0.
    """
    if stage not in ('normal','anchor_cf','anchor_control'):raise ValueError(stage)
    v=b['future_valid'].float()
    y=torch.where(v.bool(),b['y_norm'].float(),torch.zeros_like(b['y_norm'].float()))
    p_full=targets['p_full'].detach().float()
    views=[out]+(list(other_views or []) if stage=='anchor_cf' else [])
    individual=[];diagnostics=[]
    for o in views:
        l,d=_anchor_single(o,y,v,p_full,cfg,stage,float(std))
        individual.append(l);diagnostics.append(d)
    total=torch.stack(individual).mean()
    details={k:torch.stack([d[k].detach() for d in diagnostics]).mean()
             for k in diagnostics[0]}
    details['num_mask_views']=torch.tensor(float(len(views)),device=y.device)
    if (stage=='anchor_cf' and targets.get('dose_p') is not None
            and out.get('dose_psi') is not None):
        # Mechanism B on the primary view, whose mask defines the rungs.
        dose_resp=dose_response_loss(out,targets['dose_p'],v)
        total=total+float(cfg['training'].get('dose_response_weight',0.04))*dose_resp
        details['dose_response']=dose_resp.detach()
    if stage=='anchor_cf' and len(views)>1:
        # Compare fault-induced Teacher errors AND real Student candidate gains.
        t=float(cfg['training'].get('cf_error_temperature',0.35))
        teacher_ranks=[];anchor_ranks=[]
        for a in range(len(views)):
            for bidx in range(a+1,len(views)):
                aa=views[a];bb=views[bidx]
                teacher_a=((aa['backbone'].detach()-y).abs()-(p_full-y).abs()).detach()
                teacher_b=((bb['backbone'].detach()-y).abs()-(p_full-y).abs()).detach()
                target_teacher=torch.sigmoid(((teacher_a-teacher_b)/t).clamp(-20,20))
                logit_teacher=((aa['cf_teacher_gain']-bb['cf_teacher_gain'])/t).clamp(-40,40)
                teacher_ranks.append(masked_mean(F.binary_cross_entropy_with_logits(
                    logit_teacher.float(),target_teacher,reduction='none'),v))
                actual_a=actual_anchor_gain(aa['normal_prediction'],aa['normal_cf_candidate'],y)
                actual_b=actual_anchor_gain(bb['normal_prediction'],bb['normal_cf_candidate'],y)
                target_anchor=torch.sigmoid(((actual_a-actual_b)/t).clamp(-20,20))
                logit_anchor=((aa['anchor_gain_pred']-bb['anchor_gain_pred'])/t).clamp(-40,40)
                anchor_ranks.append(masked_mean(F.binary_cross_entropy_with_logits(
                    logit_anchor.float(),target_anchor,reduction='none'),v))
        t_loss=torch.stack(teacher_ranks).mean()
        a_loss=torch.stack(anchor_ranks).mean()
        total+=float(cfg['training'].get('cf_error_rank_weight',0.02))*t_loss
        total+=float(cfg['training'].get('cf_anchor_rank_weight',0.02))*a_loss
        details.update(teacher_mask_rank=t_loss.detach(),anchor_mask_rank=a_loss.detach())
    if stage=='anchor_cf' and targets.get('latent_cf_target') is not None:
        # Same hidden teacher representation only while prefix is frozen.
        diff=F.smooth_l1_loss(out['cf_latent_delta'].float(),
                              targets['latent_cf_target'].detach().float(),reduction='none').mean(-1)
        coverage=b['mask'].float().mean(1,keepdim=True)
        latent= (diff*(.2+.8*coverage)).mean()
        total+=float(cfg['training'].get('cf_latent_weight',0.01))*latent
        details['cf_latent']=latent.detach()
    return total,details
