from __future__ import annotations
import torch
from torch.nn import functional as F

def masked_mean(x,mask):
    # Never convert NaN/Inf predictions into apparently perfect (zero) loss.
    # The old nan_to_num made an invalid validation epoch score exactly 0.0.
    selected=x.masked_select(mask.bool().expand_as(x))
    if not selected.numel():
        return x.new_zeros(())
    if not torch.isfinite(selected).all():
        raise FloatingPointError('Nonfinite value at an actively supervised position')
    return selected.mean()

def losses(model,batch,out,memory,phase,mean,std,cfg):
    valid=batch['future_valid'].bool();truth=batch['y_raw'].float();scale=max(float(std),1e-5)
    pred=out['prediction']*std+mean
    abs_err=(pred-truth).abs();sq=((pred-truth)/scale).square()
    direct=valid&batch['direct_mask'].bool()[:,None,:]
    forecast=masked_mean(abs_err/scale,valid)
    missing_forecast=masked_mean(abs_err/scale,direct)
    tail=masked_mean(sq,valid)
    retrieval=forecast*0
    if phase=='retrieval':
        # Paired full/incomplete keys; only train truth is used for contrastive supervision.
        ctx=model.regions(batch['x_full'].float(),batch['raw_valid'].bool())
        enc=model.retrieval.encoder
        feats=enc._features(ctx['anchor_values'],ctx['anchor_mask'],torch.stack([torch.zeros_like(ctx['coverage']),ctx['coverage']],-1))
        paired=torch.nn.functional.normalize(enc.k(feats),dim=-1)
        qry=out['retrieval_query']
        paired_loss=1-(qry*paired).sum(-1)
        retrieval=paired_loss.mean()
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
    if phase=='retrieval':
        total= w['retrieval']*retrieval + .05*state
    elif phase=='recovery':
        total=w['state']*state+w['innovation']*innov +.1*forecast
    else:
        total=(forecast + w['direct']*missing_forecast + w['tail']*tail + w['state']*state+w['innovation']*innov+w['gate']*ctrl)
    return {'total':total,'forecast':forecast.detach(),'direct':missing_forecast.detach(),'tail':tail.detach(),'retrieval':retrieval.detach(),'state':state.detach(),'innovation':innov.detach(),'gate':out['final_gate'].mean().detach()}
