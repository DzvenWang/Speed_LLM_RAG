"""Mask-safe independent response, forecast, paired and incremental utility losses."""
import torch
from torch.nn import functional as F
from .model import bounded


def masked_mean(t,valid):
    if t.shape!=valid.shape:raise ValueError('Loss/valid mask shape mismatch')
    safe=torch.where(valid.bool(),t.float(),torch.zeros_like(t.float()))
    if not torch.isfinite(safe).all():raise RuntimeError('Loss on observed elements contains nonfinite values')
    return safe.sum()/valid.float().sum().clamp_min(1.)


def mae(pred,y,valid):
    return masked_mean((pred.float()-y.float()).abs(),valid)


def huber(pred,target,valid):
    return masked_mean(F.smooth_l1_loss(pred.float(),target.float(),reduction='none'),valid)


def response_loss(output,batch,paired=None,weights=None):
    valid=batch['future_valid'].bool()
    loss=huber(output['cf_response'],batch['cf_teacher'],valid)
    damage=huber(output['damage_prediction'],batch['damage_teacher'],valid)
    logvar=output['cf_logvar'].float()
    error=(output['cf_response'].float()-batch['cf_teacher'].float()).square()
    nll=masked_mean(.5*(torch.exp(-logvar)*error+logvar),valid)
    pair=loss.new_zeros(())
    if paired is not None:
        if not torch.equal(batch['query_id'],paired['batch']['query_id']):
            raise RuntimeError('Paired supervision uses different query identities')
        # D_a-D_b = B_b-B_a; no real future is needed for this relation.
        p=output['cf_response'].float()-paired['output']['cf_response'].float()
        target=(paired['output']['base'].float()-output['base'].float()).detach()
        pair=huber(p,target,valid & paired['batch']['future_valid'].bool())
    return dict(response=loss,damage=damage,uncertainty=nll,paired=pair)


def utility_target(output,y,bound):
    with torch.no_grad():
        r=output['normal_raw'].float();d=output['cf_response'].float();base=output['base'].float()
        normal=base+bounded(r,bound)
        proposal=base+bounded(r+d,bound)
        return (normal-y.float()).abs()-(proposal-y.float()).abs()


def stage_loss(stage,out,b,mode,bound,loss_cfg,paired=None):
    valid=b['future_valid'].bool()
    y=b['y'].float()
    if stage=='residual':
        target=b['rref'] if mode=='decomposed' else (y-out['base'].detach())
        lR=huber(out['normal_raw'],target,valid)
        # Reference Residual must not learn masked error via masked forecast loss.
        forecast=mae(out['normal_prediction'],y,valid) if mode!='decomposed' else lR.detach()*0
        return lR+loss_cfg['residual_forecast']*forecast,dict(residual=float(lR.detach()),forecast=float(forecast.detach()))
    if stage=='cf_response':
        losses=response_loss(out,b,paired)
        total=(loss_cfg['response']*losses['response']+loss_cfg['damage']*losses['damage']+
               loss_cfg['uncertainty']*losses['uncertainty']+loss_cfg['paired']*losses['paired'])
        return total,{k:float(v.detach()) for k,v in losses.items()}
    if stage in ('utility','cf_joint'):
        gain=utility_target(out,y,bound)
        ul=huber(out['cf_utility'],gain,valid)
        fl=mae(out['prediction'],y,valid)
        total=loss_cfg['forecast']*fl+loss_cfg['utility']*ul
        parts=dict(forecast=float(fl.detach()),utility=float(ul.detach()))
        if paired is not None:
            pb=paired['batch'];po=paired['output']
            if not torch.equal(b['query_id'],pb['query_id']):
                raise RuntimeError('Utility cross-mask comparison must use same TRAIN Query')
            # Query-level CF utility under two failure interventions.
            # This is NOT candidate k utility; historical v9.2 maintains that separately.
            actual_difference=(gain-utility_target(po,pb['y'],bound)).detach()
            predicted_difference=out['cf_utility'].float()-po['cf_utility'].float()
            cross_utility=huber(predicted_difference,actual_difference,
                                valid & pb['future_valid'].bool())
            total+=loss_cfg.get('cross_mask_utility',0.0)*cross_utility
            parts['cross_mask_utility']=float(cross_utility.detach())
        if stage=='cf_joint' and 'cf_teacher' in b:
            response=response_loss(out,b,paired)
            total+=(loss_cfg['response_joint']*response['response']+
                    loss_cfg['paired_joint']*response['paired'])
            parts.update(response=float(response['response'].detach()),paired=float(response['paired'].detach()))
        return total,parts
    raise ValueError(f'Unknown training stage {stage}')
