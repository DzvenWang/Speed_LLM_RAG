"""Forecast-oriented staged losses. Training targets NEVER enter model forward."""
from __future__ import annotations
import torch
from torch.nn import functional as F


def masked_mean(x,mask):
    m=mask.expand_as(x).bool()
    if not m.any():return x.sum()*0
    v=x[m]
    if not torch.isfinite(v).all():raise FloatingPointError('Nonfinite supervised positions')
    return v.mean()


def forecast_validation_loss(batch,out,mean,std):
    valid=batch['future_valid'].bool()
    pred=out['prediction']*std+mean
    mae=masked_mean((pred-batch['y_raw'].float()).abs()/max(std,1e-5),valid)
    return {'total':mae,'forecast':mae}


def losses(model,batch,out,memory,phase,mean,std,cfg):
    valid=batch['future_valid'].bool()
    truth=batch['y_norm'].float().permute(0,2,1) # [B,N,P]
    if cfg['experiment'].get('training_mode')=='no_memory':
        return forecast_validation_loss(batch,out,mean,std)
    retrieved=out['retrieval_evidence']
    probe=retrieved['probe_predictions'] # [B,N,L,P]
    candidates=retrieved['shortlist_valid']
    valid_node=valid.permute(0,2,1)[:,:,None,:]
    candidate_mae=((probe-truth[:,:,None,:]).abs()*valid_node).sum(-1)/valid_node.sum(-1).clamp_min(1)
    base=out['base_prediction'].permute(0,2,1)
    base_mae=((base-truth).abs()*valid.permute(0,2,1)).sum(-1)/valid.permute(0,2,1).sum(-1).clamp_min(1)
    utility_teacher=(base_mae[:,:,None]-candidate_mae).detach()
    # Only valid, earlier historical futures can form teacher labels.
    masked_label=candidates & valid_node.any(-1)
    zero=probe.sum()*0
    forecast=masked_mean((out['prediction']-batch['y_norm']).abs(),valid)
    if phase=='evidence':
        # Train candidate forecast probe & evidence adapter BEFORE utility ranking.
        # Hard-minimum alone would overfit; use robust mean with a soft best term.
        good=candidate_mae.masked_fill(~masked_label,10.0)
        best=good.min(-1).values
        eligible=masked_label.any(-1)
        ave=masked_mean(candidate_mae,masked_label)
        best_mean=masked_mean(best,eligible)
        training_objective=.7*ave+.3*best_mean
        return {'total':training_objective,'forecast':forecast.detach(),
                'candidate_probe_MAE':ave.detach(),'candidate_best_MAE':best_mean.detach(),
                'utility_positive_fraction':masked_mean((utility_teacher>0).float(),masked_label).detach()}
    if phase=='retrieval':
        # Node-level forecast utility, pairwise ranking and regret.
        logits=retrieved['shortlist_scores']
        util=retrieved['shortlist_utility']
        teacher=utility_teacher.clamp(-3,3)
        reg=masked_mean(F.smooth_l1_loss(util,teacher,reduction='none'),masked_label)
        pair=teacher[:,:,:-1]-teacher[:,:,1:]
        pred_pair=logits[:,:,:-1]-logits[:,:,1:]
        good_pair=masked_label[:,:,:-1] & masked_label[:,:,1:] & (pair.abs()>.005)
        pair_loss=masked_mean(F.softplus(-pred_pair*pair.sign()),good_pair)
        probs=torch.softmax(logits.masked_fill(~masked_label,-1e4),-1)*masked_label.float()
        probs=probs/probs.sum(-1,keepdim=True).clamp_min(1e-6)
        best=teacher.masked_fill(~masked_label,-1e4).max(-1).values
        chosen=(probs*teacher).sum(-1)
        active=masked_label.any(-1)
        regret=masked_mean((best-chosen).clamp_min(0),active)
        # Mask-consistent same-episode key alignment gives Key Encoder gradients.
        paired=model.retrieval.encoder.encode_key_examples(batch['x_full'].float(),
                                    batch['raw_valid'].bool(),model.regions)
        query=retrieved['region_query']
        consistency=(1-(query*paired).sum(-1)).mean()
        coarse_benefit=coarse_benefit_contrastive_loss(retrieved,utility_teacher,masked_label,
                                                       model.regions.owner)
        total=.7*reg+.3*pair_loss+.4*regret+.2*coarse_benefit+.05*consistency
        return {'total':total,'forecast':forecast.detach(),'utility_regression':reg.detach(),
                'utility_pairwise':pair_loss.detach(),'utility_regret':regret.detach(),
                'utility_positive_fraction':masked_mean((teacher>0).float(),masked_label).detach(),
                'query_consistency':consistency.detach(),
                'coarse_benefit_contrastive':coarse_benefit.detach()}
    # Integration and control: actual future MAE, with candidate auxiliary loss.
    candidate=masked_mean((out['candidate_prediction']-batch['y_norm']).abs(),valid)
    directly_masked=valid & batch['direct_mask'].bool()[:,None,:]
    direct=masked_mean((out['prediction']-batch['y_norm']).abs(),directly_masked)
    correction=masked_mean(F.smooth_l1_loss(out['candidate_prediction']-out['base_prediction'],
                                torch.zeros_like(out['candidate_prediction']),reduction='none'),valid)
    # Additional candidate loss counters the easy all-null/zero-gate solution.
    total=forecast+.2*candidate+.2*direct+.002*correction
    return {'total':total,'forecast':forecast.detach(),'candidate':candidate.detach(),
            'direct':direct.detach(),'correction_energy':correction.detach(),
            'retrieval_null_fraction':retrieved['null_weight'].mean().detach()}


def coarse_benefit_contrastive_loss(retrieved,benefit,valid,owner,temperature=0.25):
    """Forecast-utility supervision of differentiable selected coarse scores.

    A candidate gets credit when its historical ID is beneficial to a node in
    that graph region. This lets the *region query* learn beyond reconstruction
    or same-episode identity, even though the discrete topk IDs are non-grad.
    """
    scores=retrieved['coarse_scores']  # [B,G,C], with live query gradients
    coarse= retrieved['coarse_ids']
    shortlist=retrieved['shortlist_ids']
    b,g,c=coarse.shape
    target=scores.new_zeros(b,g,c)
    region_active=torch.zeros(b,g,dtype=torch.bool,device=scores.device)
    for rid in range(g):
        node_mask=(owner==rid)
        if not node_mask.any():continue
        chosen=shortlist[:,node_mask,:]
        benefit_group=benefit[:,node_mask,:].clamp_min(0)*valid[:,node_mask,:].float()
        same=(chosen[...,None]==coarse[:,rid,None,None,:])
        value=(same.float()*benefit_group[...,None]).sum((1,2))
        target[:,rid]=value
        region_active[:,rid]=value.sum(-1)>1e-6
    if not region_active.any():return scores.sum()*0
    scaled=scores/temperature
    target=target/target.sum(-1,keepdim=True).clamp_min(1e-8)
    logp=F.log_softmax(scaled,-1)
    return -(target*logp).sum(-1)[region_active].mean()
