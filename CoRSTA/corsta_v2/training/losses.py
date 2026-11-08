"""Raw-unit forecast-first objective with distinct CF and complete-view residual semantics."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from corsta_v2.util import masked_mean
from corsta_v2.counterfactual.error_contrast import error_contrast_losses,candidate_gain


def compute_loss(out,b,t,stage,mean,std,cfg,contrast=None):
    c=cfg['training'];v=b['future_valid'].float()
    y=torch.where(v.bool(),b['y_norm'].float(),torch.zeros_like(b['y_norm'].float()))
    pred=out['prediction'].float()
    error=(pred-y)*std
    mae=masked_mean(error.abs(),v)
    rmse=torch.sqrt(masked_mean(error.square(),v)+1e-8)
    total=mae+float(c['forecast_rmse_weight'])*rmse
    details={'mae':mae.detach(),'rmse':rmse.detach(),'forecast':total.detach()}
    if stage in ('cf','joint'):
        cf_target=t['cf_target'].detach()
        # Preserve raw CF response supervision even when learned action gate
        # suppresses a candidate. Otherwise benefit learning could collapse CF.
        cf_error=F.smooth_l1_loss(out['delta_cf_candidate'].float(),torch.where(v.bool(),cf_target.float(),torch.zeros_like(cf_target.float())),reduction='none')
        tau=0.35 # normalized-unit, teacher-benefit weighting only in TRAIN
        floor=float(c['cf_utility_floor'])
        utility=t['cf_utility'].detach()
        weight=floor+(1-floor)*torch.sigmoid(utility/tau)
        cf_loss=masked_mean(cf_error*weight,v)
        total=total+float(c['cf_response_weight'])*cf_loss
        details['cf_response']=cf_loss.detach()
        # A single-view error-gain target is available in every CF/JOIN batch;
        # multi-view error ranking is an additional, strictly TRAIN-only loss.
        gain_loss=masked_mean(F.smooth_l1_loss(
            out['cf_teacher_gain'].float(),t['cf_utility'].detach().float(),
            reduction='none'),v)
        total+=float(c.get('cf_error_gain_weight',0.03))*gain_loss
        details['cf_teacher_error_gain']=gain_loss.detach()
        # Actual candidate benefit differs from the hypothetical teacher gain.
        # The action head is trained with a label based on the *current*
        # detached student candidate, never future labels at inference.
        y_for_gate=y.detach()
        action_util=candidate_gain(out['backbone'],out['cf_raw_prediction'],y_for_gate)
        temperature=float(c.get('cf_error_temperature',0.35))
        action_target=torch.sigmoid((action_util/temperature).clamp(-20,20))
        accept_loss=masked_mean(F.binary_cross_entropy_with_logits(
            out['cf_accept_logit'].float(),action_target,reduction='none'),v)
        total+=float(c.get('cf_acceptance_weight',0.02))*accept_loss
        details['cf_action_acceptance']=accept_loss.detach()
        if contrast is not None:
            extra,extra_details=error_contrast_losses(out,contrast,b,t,cfg)
            total=total+extra
            details.update(extra_details)
        if t.get('latent_cf_target') is not None:
            diff=F.smooth_l1_loss(out['cf_latent_delta'].float(),t['latent_cf_target'].float(),reduction='none').mean(-1)
            # Do not require complete-view features to be fully identifiable under severe outages.
            coverage=b['mask'].float().mean(1,keepdim=True)
            latent_w=.2+.8*coverage
            cf_lat=diff.mul(latent_w).mean()
            total=total+float(c['cf_latent_weight'])*cf_lat
            details['cf_latent']=cf_lat.detach()
    if stage in ('residual','joint'):
        delta=out['delta_residual'].float()
        semantic=t['residual_target'].detach().float()
        semantic=torch.where(v.bool(),semantic,torch.zeros_like(semantic))
        rloss=masked_mean(F.smooth_l1_loss(delta,semantic,reduction='none'),v)
        total=total+float(c['residual_semantic_weight'])*rloss
        details['residual_semantic']=rloss.detach()
    if stage=='control':
        # Actual utility belongs to current candidate (not full-view teacher).
        p1=out['cf_prediction'].detach().float();p2=out['cf_residual_prediction'].detach().float()
        utility=((p1-y).abs()-(p2-y).abs()).detach()
        gate=out['gate_residual']
        target=(utility>0).float()
        bce=masked_mean(F.binary_cross_entropy(gate.float().clamp(1e-5,1-1e-5),target,reduction='none'),v)
        benefit=out['control']['benefit_logit']
        benefit_loss=masked_mean(F.smooth_l1_loss(benefit.float(),utility,reduction='none'),v)
        total=total+.03*bce+.02*benefit_loss
        details.update(benefit_classification=bce.detach(),benefit_regression=benefit_loss.detach())
    return total,details


def compute_metrics(out,b,mean,std):
    y=b['y_raw'].float();v=b['future_valid'].float()
    result={}
    fields=([('backbone','backbone'),('normal','normal_prediction'),
             ('normal_cf_candidate','normal_cf_candidate'),('final','prediction')]
            if 'normal_prediction' in out else
            [('backbone','backbone'),('cf','cf_prediction'),
             ('cf_residual','cf_residual_prediction'),('final','prediction')])
    for name,key in fields:
        error=out[key].float()*std+mean-y
        ab=(error.abs()*v).sum().item();sq=(error.square()*v).sum().item();count=v.sum().item()
        result[name]=(ab,sq,count)
    return result
