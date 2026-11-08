"""Train-only forecasting utility, marginal contribution and alignment oracles.

The future target is never part of model.forward; all teacher tensors are masked
by future validity BEFORE any reduction. No pseudo-label may read TEST future.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F
from ..v9.model import utility_targets, masked_mae, paired_cross_failure_loss


def marginal_targets(out,target,future_valid,valid,blocks=3):
    """Leave-one-out contribution with renormalization over remaining K and Null."""
    c=out['candidate_cf'].float();b,k,h,n=c.shape;span=h//blocks
    if h%blocks:raise ValueError('Horizon not divisible by blocks')
    w=out['cf_weights'][:,:k].detach().float()   # [B,K,G,N]
    with torch.no_grad():
        m=out['cf_raw'].float().reshape(b,blocks,span,n)
        wc=w[:,:,:,None]*c.reshape(b,k,blocks,span,n)
        denominator=1.-w[:,:,:,None]
        leave=torch.where(denominator>1e-6,(m[:,None]-wc)/denominator.clamp_min(1e-6),
                          torch.zeros_like(wc))
        f=lambda x:out['bound']*torch.tanh(x/out['bound'])
        nraw=out['normal_raw'].float().reshape(b,blocks,span,n)
        base=out['base'].float().reshape(b,blocks,span,n)
        pred_without=base[:,None]+f(nraw[:,None]+leave)
        pred_all=out['prediction'].detach().float().reshape(b,blocks,span,n)
        y=target.float().reshape(b,blocks,span,n)
        fv=future_valid.bool().reshape(b,blocks,span,n)
        safe_y=torch.where(fv,y,torch.zeros_like(y))
        improvement=(pred_without-safe_y[:,None]).abs()-(pred_all[:,None]-safe_y[:,None]).abs()
        improvement=torch.where(fv[:,None],improvement,torch.zeros_like(improvement))
        count=fv.sum(2)
        targets=improvement.sum(3)/count[:,None].clamp_min(1)
        is_valid=(count[:,None]>0)&valid[:,:,None,None].bool()
        targets=torch.where(is_valid,targets,torch.zeros_like(targets))
    return targets,is_valid


def oracle_alignment(out,cf,target,future_valid,valid,blocks=3,scales=(-1.5,-1.,-.5,0.,.5,1.,1.5)):
    """Discrete TRAIN-only coefficient argmin; ignore indistinguishable CF candidates."""
    with torch.no_grad():
        b,k,h,n=cf.shape;span=h//blocks
        if h%blocks:raise ValueError('Horizon/block mismatch')
        candidate=cf.float().reshape(b,k,blocks,span,n)
        y=target.float().reshape(b,blocks,span,n)
        fv=future_valid.bool().reshape(b,blocks,span,n)
        safe_y=torch.where(fv,y,torch.zeros_like(y))
        nraw=out['normal_raw'].detach().float().reshape(b,blocks,span,n)
        base=out['base'].detach().float().reshape(b,blocks,span,n)
        losses=[]
        for alpha in scales:
            proposal=base[:,None]+out['bound']*torch.tanh((nraw[:,None]+float(alpha)*candidate)/out['bound'])
            err=torch.where(fv[:,None],(proposal-safe_y[:,None]).abs(),torch.zeros_like(proposal))
            losses.append(err.sum(3)/fv.sum(2)[:,None].clamp_min(1))
        loss=torch.stack(losses,0)
        idx=loss.argmin(0)
        alpha=torch.tensor(scales,device=cf.device,dtype=cf.dtype)[idx]
        span_loss=loss.amax(0)-loss.amin(0)
        support=(fv.sum(2)[:,None]>0)&valid[:,:,None,None].bool()&(span_loss>1e-5)
        return alpha,support,dict(signed_best=loss.amin(0),zero_error=loss[list(scales).index(0.)])


def cf_training_loss(out,target,future_valid,candidate_valid,cfg,stage,teacher_cf=None,raw_cf=None):
    if stage=='normal' or 'cf_utility' not in out:
        forecast=masked_mae(out['prediction'].float(),target.float(),future_valid)
        return forecast,{'forecast':forecast.detach()}
    pred=out['prediction'].float();target=target.float();fv=future_valid.bool()
    total=masked_mae(pred,target,fv)
    terms={'forecast':total.detach()}
    u,uv=utility_targets(out['base'].float(),out['normal_raw'].float(),out['candidate_cf'].float(),
                target,candidate_valid,out['bound'],out['cf_utility'].shape[2],future_valid=fv,return_valid=True)
    if stage in ('utility','joint','paired'):
        loss=F.smooth_l1_loss(out['cf_utility'].float(),u,reduction='none')
        reg=torch.where(uv,loss,torch.zeros_like(loss)).sum()/uv.sum().clamp_min(1)
        total=total+cfg.get('utility_weight',.02)*reg;terms['utility']=reg.detach()
        truth=torch.cat([u.masked_fill(~uv,-1e6),torch.full_like(u[:,:1],1e-6)],1).argmax(1)
        valid_location=fv.reshape(fv.shape[0],u.shape[2],-1,fv.shape[-1]).any(2)
        prob=out['cf_weights'].float().clamp_min(1e-9)
        selected=prob.gather(1,truth[:,None]).squeeze(1)
        rank=-(selected.log()*valid_location).sum()/valid_location.sum().clamp_min(1)
        total=total+cfg.get('rank_weight',.01)*rank;terms['ranking']=rank.detach()
    if stage in ('joint','paired') and cfg.get('marginal_weight',0)>0:
        target_margin,valid_margin=marginal_targets(out,target,fv,candidate_valid,out['cf_utility'].shape[2])
        m_loss=F.smooth_l1_loss(out['marginal_utility'].float(),target_margin,reduction='none')
        marg=torch.where(valid_margin,m_loss,torch.zeros_like(m_loss)).sum()/valid_margin.sum().clamp_min(1)
        total=total+cfg['marginal_weight']*marg;terms['marginal']=marg.detach()
    if stage in ('align','joint','paired') and raw_cf is not None and cfg.get('alignment_weight',0)>0:
        oracle,good,_=oracle_alignment(out,raw_cf,target,fv,candidate_valid,out['cf_utility'].shape[2])
        align=F.smooth_l1_loss(out['alignment'].float(),oracle.float(),reduction='none')
        align=torch.where(good,align,torch.zeros_like(align)).sum()/good.sum().clamp_min(1)
        total=total+cfg['alignment_weight']*align;terms['alignment']=align.detach()
    if stage in ('joint','paired') and teacher_cf is not None and teacher_cf.numel()>0 and cfg.get('full_info_weight',0)>0:
        # Predict the remaining benefit of full sensing relative to the fixed Normal anchor.
        teacher=out['base'].float()+teacher_cf.float()
        gain=(out['normal_prediction'].float()-target).abs()-(teacher-target).abs()
        gain=torch.where(fv,gain,torch.zeros_like(gain))
        b,h,n=gain.shape;g=out['recoverability'].shape[1];span=h//g
        count=fv.reshape(b,g,span,n).sum(2)
        label=gain.reshape(b,g,span,n).sum(2)/count.clamp_min(1)
        loss=F.smooth_l1_loss(out['recoverability'].float(),label.detach(),reduction='none')
        full=torch.where(count>0,loss,torch.zeros_like(loss)).sum()/(count>0).sum().clamp_min(1)
        total=total+cfg['full_info_weight']*full;terms['fullinfo']=full.detach()
    return total,terms
