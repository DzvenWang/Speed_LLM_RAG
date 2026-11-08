"""Forecast-first v5 objectives; labels exclusively from train/validation futures.

* Positive gain and harm are distinct nonnegative targets.
* C1 utility affects inference and is supervised by incremental utility.
* Paired negatives are supervised only when empirically worse (no fake labels).
* Stage-two marginal targets use the INITIAL mixture used by the reranker.
* Controller losses apply in controller AND joint stages.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def mean_valid(a,valid):
    v=valid.float()
    return (a.float()*v).sum()/v.sum().clamp_min(1.)


def compute_losses(o,b,mean,std,cfg,stage='c12',auxiliary=False):
    valid=b['future_valid'].bool();y=b['y_norm'].float();base=o['base_prediction'].detach().float()
    pred=o['prediction'].float();err=pred*std+mean-b['y_raw'].float()
    mae=mean_valid(err.abs(),valid)
    rmse=torch.sqrt(mean_valid(err.square(),valid)+1e-8)
    opt=cfg.get('v5',{})
    total=mae+float(cfg['training'].get('rmse_weight',.05))*rmse
    stats={'mae':mae,'rmse_batch':rmse,'forecast':mae}
    if auxiliary:
        stats['total']=total
        return total,{k:v.detach() for k,v in stats.items()}

    if stage in ('c12','joint'):
        c=o['candidate_residuals'].float();bound=float(o['residual_bound'])
        avail=o['candidate_available'].bool() & valid[:,None]
        proposal_c=bound*torch.tanh(c/bound)
        signed=((base[:,None]-y[:,None]).abs()-(base[:,None]+proposal_c-y[:,None]).abs()).detach()
        pos=signed.clamp_min(0);harm=(-signed).clamp_min(0)
        gain_pred=o['candidate_transfer_gain'].float();harm_pred=o['candidate_transfer_harm'].float()
        if avail.any():
            indiv=F.smooth_l1_loss(gain_pred[avail],pos[avail])+F.smooth_l1_loss(harm_pred[avail],harm[avail])
        else:indiv=mae*0
        true_logits=torch.cat((signed.masked_fill(~avail,-1e4),torch.zeros_like(base)[:,None]),1)/max(float(opt.get('label_temperature',.25)),1e-4)
        soft=torch.softmax(true_logits,dim=1)
        ranking=mean_valid(-(soft*F.log_softmax(o['candidate_logits'].float(),dim=1)).sum(1),valid)

        normal=o['normal_component'].float();increment=o['cf_increment'].float()
        normal_pred=bound*torch.tanh(normal/bound)
        cf_pred=bound*torch.tanh((normal+increment)/bound)
        cf_target=((base[:,None]+normal_pred-y[:,None]).abs()-
                   (base[:,None]+cf_pred-y[:,None]).abs()).detach()
        if avail.any():
            cf_utility=F.smooth_l1_loss(o['cf_incremental_utility'].float()[avail],cf_target[avail])
            # CUDA autocast forbids BCE on a sigmoid probability. Supervise the
            # raw C1 logit with the numerically stable, autocast-safe loss.
            cf_bce=F.binary_cross_entropy_with_logits(
                o['cf_gain_logit'].float()[avail],
                (cf_target[avail]>0).float())
        else:cf_utility=cf_bce=mae*0
        wrong_pair=mae*0
        if 'cf_wrong_increment' in o and avail.any():
            wrong=bound*torch.tanh((normal+o['cf_wrong_increment'].float())/bound)
            util_wrong=((base[:,None]+normal_pred-y[:,None]).abs()-
                       (base[:,None]+wrong-y[:,None]).abs()).detach()
            # Symmetric preference: whichever pairing gives higher realized
            # incremental forecasting gain must have the higher learned score.
            # Never assume the semantically matched CF is always beneficial.
            utility_gap=cf_target-util_wrong
            informative=avail & (utility_gap.abs()>float(opt.get('pair_ground_truth_margin',.005)))
            if informative.any():
                correct_score=o['cf_incremental_utility'].float()
                wrong_score=o['cf_wrong_utility_pred'].float()
                preference=utility_gap.detach().sign()
                wrong_pair=F.softplus((wrong_score[informative]-correct_score[informative])
                                      *preference[informative]+
                                      float(opt.get('pair_ranking_margin',.05))).mean()

        # Target for second ranking pass is the increment relative to the
        # initial mixture and initial leave-one-out, never the final mixture.
        initial=o['marginal_reference_proposal'].float().detach()
        loo=o['candidate_loo_proposal'].float().detach()
        marginal_target=((base[:,None]+loo-y[:,None]).abs()-
                         (base[:,None]+initial[:,None]-y[:,None]).abs()).detach()
        marginal=(F.smooth_l1_loss(o['candidate_marginal_utility_pred'].float()[avail],
                                  marginal_target[avail]) if avail.any() else mae*0)
        total=(total+float(opt.get('utility_weight',.05))*indiv+
               float(opt.get('rank_weight',.03))*ranking+
               float(opt.get('cf_utility_weight',.02))*cf_utility+
               float(opt.get('cf_gain_bce_weight',.005))*cf_bce+
               float(opt.get('cf_pair_weight',.005))*wrong_pair+
               float(opt.get('marginal_utility_weight',.01))*marginal)
        stats.update(utility=indiv,ranking=ranking,cf_incremental_utility=cf_utility,
                     cf_gain_bce=cf_bce,cf_pair=wrong_pair,marginal_utility=marginal,
                     cf_increment_abs=mean_valid(increment.abs().mean(1),valid),
                     support_logit_abs=mean_valid(o['support_logits_delta'].abs().mean(1),valid),
                     support_utility_abs=mean_valid(o['support_utility_delta'].abs().mean((1,4)),valid))

    if stage in ('controller','joint'):
        scales=o['scale_choices'].float()
        proposal=o['ungated_correction'].detach().float()
        poss=base[:,None]+scales[None,:,None,None]*proposal[:,None]
        realized=((base[:,None]-y[:,None]).abs()-(poss-y[:,None]).abs()).detach()
        mask=valid[:,None].expand_as(realized)
        positive_target=realized.clamp_min(0);harm_target=(-realized).clamp_min(0)
        scale_pred=o['scale_utility_logits'].float().movedim(-1,1)
        utility_loss=mean_valid(F.smooth_l1_loss(scale_pred,realized,reduction='none'),mask)
        gain_loss=mean_valid(F.smooth_l1_loss(o['scale_positive_gain'].movedim(-1,1).float(),
                                             positive_target,reduction='none'),mask)
        harm_loss=mean_valid(F.smooth_l1_loss(o['scale_positive_harm'].movedim(-1,1).float(),
                                             harm_target,reduction='none'),mask)
        harm_bce=mean_valid(F.binary_cross_entropy_with_logits(
            o['scale_harm_logits'].movedim(-1,1).float(),(realized<0).float(),
            reduction='none'),mask)
        ctrl=utility_loss+.25*(gain_loss+harm_loss)+.05*harm_bce
        total=total+float(opt.get('scale_utility_weight',.05))*ctrl
        # CFRM and SCRT allocate a full bounded proposal into two exactly
        # additive sources. Supervise the correction value of using Normal,
        # CF, both and the non-additive utility interaction. The first two
        # labels are valid even where the components cancel in the mixture.
        n=o['proposed_normal'].detach().float()
        cf=o['proposed_cf'].detach().float()
        e0=(base-y).abs()
        un=(e0-(base+n-y).abs()).detach()
        uf=(e0-(base+cf-y).abs()).detach()
        uj=(e0-(base+n+cf-y).abs()).detach()
        interaction=uj-un-uf
        utility_targets=torch.stack((un,uf,interaction),dim=-1)
        u_loss=mean_valid(F.smooth_l1_loss(
            o['component_utility_pred'].float(),utility_targets,reduction='none').mean(-1),valid)
        harm_targets=torch.stack((un<0,uf<0),dim=-1).float()
        harm_components=mean_valid(F.binary_cross_entropy_with_logits(
            o['component_harm_logits'].float(),harm_targets,reduction='none').mean(-1),valid)
        total=total+float(opt.get('component_utility_weight',.10))*u_loss+float(
            opt.get('component_harm_weight',.02))*harm_components
        # True outcome of the actually SELECTED Normal/CF mixture BEFORE
        # optional calibration rejection. Component harm labels alone are
        # insufficient: two individually good changes may be jointly harmful.
        selected=o['pre_rejection_correction'].float()
        joint_harm_target=((base+selected-y).abs()>(base-y).abs()).float().detach()
        joint_harm_bce=mean_valid(F.binary_cross_entropy_with_logits(
            o['joint_harm_logit'].float(),joint_harm_target,reduction='none'),valid)
        total=total+float(opt.get('joint_harm_weight',.04))*joint_harm_bce
        stats['joint_harm_bce']=joint_harm_bce
        # Spatial/Horizon regularization on the two source-specific gates.
        # Preserve genuine local deviations; regularize only abrupt temporal
        # jumps and pairwise functional-neighborhood disagreement softly.
        if 'normal_gate' in o and o['normal_gate'].shape[1]>1:
            gates=torch.stack((o['normal_gate'],o['cf_gate']),-1)
            smooth=(gates[:,1:]-gates[:,:-1]).abs().mean()
            total=total+float(opt.get('gate_temporal_smooth_weight',.001))*smooth
            stats['gate_temporal_smooth']=smooth
            # Functional-neighborhood smoothness and bounded local departure.
            neighbours=o.get('functional_neighbours')
            if neighbours is not None and neighbours.numel()>0:
                ids=neighbours.long().clamp(0,gates.shape[2]-1)
                sampled=gates[:,:,ids,:].mean(dim=3)
                smooth_space=(gates-sampled).abs().mean()
                total=total+float(opt.get('gate_spatial_smooth_weight',.0005))*smooth_space
                stats['gate_spatial_smooth']=smooth_space
            local_offset=o.get('local_intervention_deviation')
            if local_offset is not None:
                local_penalty=local_offset.float().abs().mean()
                total=total+float(opt.get('c3_local_offset_weight',.0002))*local_penalty
                stats['c3_local_offset']=local_penalty
        stats.update(scale_utility=utility_loss,scale_gain=gain_loss,scale_harm=harm_loss,
                     scale_harm_bce=harm_bce,component_utility=u_loss,component_harm=harm_components)
    stats['gate']=mean_valid(o['gate'],valid)
    stats['total']=total
    return total,{k:v.detach() for k,v in stats.items()}
