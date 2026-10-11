from __future__ import annotations
import os
import torch
from torch.nn import functional as F
from .ranking import target_recovery_ranking_loss,forecast_utility_ranking_loss

def masked_mean(x,mask):
    selected=x.masked_select(mask.bool().expand_as(x))
    if not selected.numel():return x.new_zeros(())
    if not torch.isfinite(selected).all():
        raise FloatingPointError('Nonfinite value at an actively supervised position')
    return selected.mean()

def forecast_validation_loss(batch,out,mean,std):
    """Canonical normalized forecast MAE only, used for checkpoint selection.

    Skips candidate/ranking/recovery/teacher losses when the selected
    checkpoint metric is already the forecast MAE. Does not change model forward.
    """
    valid=batch['future_valid'].bool()
    err=(out['prediction']*std+mean-batch['y_raw'].float()).abs()/max(float(std),1e-5)
    forecast=masked_mean(err,valid)
    return {'total':forecast,'forecast':forecast}


def retrieval_phase_loss(model,batch,out,memory,mean,std,cfg):
    """Equivalent retrieval-stage objective with no Teacher/second STLA pass.

    Only retrieval query, selected evidence weights and recovered historical
    state are necessary for this phase. No future target enters the model forward.
    """
    ref=out['recovered_state']
    z=ref.sum()*0
    rank=z;rank_diag={};utility=z;utility_diag={}
    if cfg.get('ranking',{}).get('enabled',True):
        rank,rank_diag=target_recovery_ranking_loss(
            batch,{'ids':out['retrieved_ids'],'valid':out['retrieved_valid_candidates'],
                   'weights':out['retrieved_weights']},memory,model.regions,
            float(cfg['ranking'].get('target_temperature',.25)))
    if cfg.get('forecast_utility',{}).get('enabled',False):
        utility,utility_diag=forecast_utility_ranking_loss(
            batch,{'ids':out['retrieved_ids'],'valid':out['retrieved_valid_candidates'],
                   'weights':out['retrieved_weights']},memory,model.regions,
            float(cfg['forecast_utility'].get('teacher_temperature',.35)))
    ctx=model.regions(batch['x_full'].float(),batch['raw_valid'].bool())
    enc=model.retrieval.encoder
    features=enc._features(ctx['anchor_values'],ctx['anchor_mask'],
        torch.stack([torch.zeros_like(ctx['coverage']),ctx['coverage']],-1))
    paired=F.normalize(enc.k(features),dim=-1)
    r=(1-(out['retrieval_query']*paired).sum(-1)).mean()
    wanted=batch['artificial_mask'].bool()&batch['raw_valid'].bool()
    state=masked_mean((ref-batch['x_full']).abs(),wanted)
    w=cfg['loss']
    total=w['retrieval']*r+w.get('ranking',.15)*rank+.05*state+cfg.get('forecast_utility',{}).get('loss_weight',.20)*utility
    return {'total':total,'retrieval':r.detach(),'ranking':rank.detach(),
            'forecast_utility_rank':utility.detach(),'state':state.detach(),
            'forecast':z.detach(),'direct':z.detach(),'tail':z.detach(),
            'candidate':z.detach(),'correction_energy':z.detach(),
            'innovation':z.detach(),'gate':z.detach(),**rank_diag,**utility_diag}

def losses(model,batch,out,memory,phase,mean,std,cfg):
    if phase=='retrieval' and cfg['experiment'].get('training_mode','full')!='no_memory':
        return retrieval_phase_loss(model,batch,out,memory,mean,std,cfg)
    valid=batch['future_valid'].bool();truth=batch['y_raw'].float();scale=max(float(std),1e-5)
    pred=out['prediction']*std+mean
    abs_err=(pred-truth).abs();sq=((pred-truth)/scale).square()
    direct=valid&batch['direct_mask'].bool()[:,None,:]
    forecast=masked_mean(abs_err/scale,valid)
    missing_forecast=masked_mean(abs_err/scale,direct)
    tail=masked_mean(sq,valid)
    # Proposal supervision is deliberately weak: the alpha=1 candidate need
    # not outperform the base, but its huge errors should not be cost-free.
    candidate_raw=out['candidate_prediction']*std+mean
    candidate_loss=masked_mean(F.smooth_l1_loss((candidate_raw-truth)/scale,
                               torch.zeros_like(candidate_raw),reduction='none'),valid)
    delta=out['candidate_prediction']-out['base_prediction']
    # A small correction-magnitude regularizer improves stability and should
    # not stop the model from learning helpful large corrections.
    correction_energy=masked_mean(F.smooth_l1_loss(delta,torch.zeros_like(delta),reduction='none'),valid)
    if cfg['experiment'].get('training_mode','full')=='no_memory':
        total=forecast+cfg['loss']['direct']*missing_forecast+cfg['loss']['tail']*tail
        zero=total.detach()*0
        return {'total':total,'forecast':forecast.detach(),'direct':missing_forecast.detach(),
                'tail':tail.detach(),'candidate':zero,'correction_energy':zero,
                'retrieval':zero,'ranking':zero,'state':zero,'innovation':zero,'gate':zero}
    retrieval=forecast*0
    ranking=forecast*0
    rank_diag={}
    if phase in ('retrieval','recovery','injection','partial','full') and cfg.get('ranking',{}).get('enabled',True):
        rank_temp=float(cfg.get('ranking',{}).get('target_temperature',.25))
        ranking,rank_diag=target_recovery_ranking_loss(
            batch,{'ids':out['retrieved_ids'],'valid':out['retrieved_valid_candidates'],
                   'weights':out['retrieved_weights']},memory,model.regions,rank_temp)

    utility_rank=forecast*0
    utility_diag={}
    if cfg.get('forecast_utility',{}).get('enabled',False) and phase in ('retrieval','recovery','injection','partial','full'):
        utility_rank,utility_diag=forecast_utility_ranking_loss(
            batch,{'ids':out['retrieved_ids'],'valid':out['retrieved_valid_candidates'],
                   'weights':out['retrieved_weights']},memory,model.regions,
            cfg['forecast_utility'].get('teacher_temperature',.35))
    if phase=='retrieval':
        ctx=model.regions(batch['x_full'].float(),batch['raw_valid'].bool())
        enc=model.retrieval.encoder
        feats=enc._features(ctx['anchor_values'],ctx['anchor_mask'],torch.stack([torch.zeros_like(ctx['coverage']),ctx['coverage']],-1))
        paired=F.normalize(enc.k(feats),dim=-1)
        qry=out['retrieval_query']
        retrieval=(1-(qry*paired).sum(-1)).mean()
    rec_mask=batch['artificial_mask'].bool()&batch['raw_valid'].bool()
    state=masked_mean((out['recovered_state']-batch['x_full']).abs(),rec_mask)
    with torch.no_grad():
        full_hidden=model.teacher_hidden(batch['x_full'].float(),batch['raw_valid'].bool(),batch['tod'],batch['dow'])
        current_hidden=model.teacher_hidden(out['supported_history'].detach(),batch['obs_mask'],batch['tod'],batch['dow'])
        dz_target=(full_hidden-current_hidden).detach()
    innov_weight=(~batch['obs_mask']).float()[...,None]
    innov=masked_mean(F.smooth_l1_loss(out['latent_innovation'],dz_target,reduction='none'),innov_weight.expand_as(dz_target))
    ctrl=(out['final_gate']**2).mean()
    w=cfg['loss']
    utility_term=cfg.get('forecast_utility',{}).get('loss_weight',.20)*utility_rank
    if phase=='retrieval':
        total=w['retrieval']*retrieval+w.get('ranking',.15)*ranking+.05*state+utility_term
    elif phase=='recovery':
        total=w['state']*state+w['innovation']*innov+.1*forecast+w.get('ranking',.15)*ranking+utility_term
    else:
        total=(forecast+w['direct']*missing_forecast+w['tail']*tail+
               w['state']*state+w['innovation']*innov+w['gate']*ctrl+
               w.get('candidate',.08)*candidate_loss+
               w.get('correction_energy',.003)*correction_energy+
               w.get('ranking',.15)*ranking+utility_term)
    return {'total':total,'ranking':ranking.detach(),'forecast_utility_rank':utility_rank.detach(),**utility_diag,**rank_diag,'forecast':forecast.detach(),'direct':missing_forecast.detach(),
            'tail':tail.detach(),'candidate':candidate_loss.detach(),
            'correction_energy':correction_energy.detach(),
            'retrieval':retrieval.detach(),'state':state.detach(),
            'innovation':innov.detach(),'gate':out['final_gate'].mean().detach()}
