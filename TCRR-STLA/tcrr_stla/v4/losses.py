"""Train-only supervised losses; test targets never enter a model forward."""
from __future__ import annotations
import torch
import torch.nn.functional as F

def masked_mean(value,mask):
    return (value.float()*mask.float()).sum()/mask.float().sum().clamp_min(1)

def valid_cvar(harm,valid,tail_fraction=0.1):
    x=harm[valid.bool()].float()
    if x.numel()==0:return harm.sum()*0
    n=max(1,int((x.numel()*tail_fraction)+0.999999))
    return torch.topk(x,k=n,largest=True,sorted=False).values.mean()

def compute_v4_losses(out,batch,mean,std,cfg,stage='joint'):
    """Full mask/horizon forecast + candidate gain/harm + tail control."""
    raw=out['prediction'].float()*std+mean
    base=out['base_prediction'].float().detach()
    y=batch['y_norm'].float()
    valid=batch['future_valid'].bool()
    if not bool(valid.any()):raise RuntimeError('No valid future labels')
    fc=masked_mean(torch.abs(raw-batch['y_raw'].float()),valid)
    rmse=torch.sqrt(masked_mean((raw-batch['y_raw'].float()).square(),valid)+1e-8)
    zero=fc.new_zeros(())
    if out.get('candidate_weights') is None:
        total=fc+float(cfg['training'].get('rmse_weight',.05))*rmse
        return total,dict(mae=fc.detach(),rmse=rmse.detach(),forecast=fc.detach(),total=total.detach())
    candidate=out['candidate_residuals'].float()
    cv=out['candidate_available'].bool()&valid[:,None]
    bound=float(out['residual_bound'])
    proposed=bound*torch.tanh(candidate/bound)
    gain_label=(base[:,None]-y[:,None]).abs()-(base[:,None]+proposed-y[:,None]).abs()
    gain_label=gain_label.detach()
    gain_pred=out['candidate_transfer_gain'].float()
    harm_pred=out['candidate_transfer_harm'].float()
    if bool(cv.any()):
        utility=F.smooth_l1_loss(gain_pred[cv],gain_label[cv])
        utility=utility+0.25*F.smooth_l1_loss(harm_pred[cv],(-gain_label[cv]).clamp_min(0))
    else:utility=zero
    # The null candidate has true gain zero and competes explicitly.
    temp=float(cfg.get('v4',{}).get('label_temperature',.25))
    true_logits=torch.cat([gain_label.masked_fill(~cv,-1e4),torch.zeros_like(base)[:,None]],1)/temp
    soft=F.softmax(true_logits,dim=1)
    logits=out['candidate_logits'].float()
    ranking=masked_mean(-(soft*F.log_softmax(logits,dim=1)).sum(1),valid)
    # Correctly paired candidates must only be preferred when their REAL
    # query-target improvement exceeds hard shuffled alternatives.
    paired=out.get('hard_shuffled_candidate')
    wrong_pred=out.get('hard_shuffled_gain_pred')
    pair=zero
    if paired is not None and wrong_pred is not None and candidate.shape[1]>1:
        wrong=bound*torch.tanh(paired.float()/bound)
        wrong_gain=((base[:,None]-y[:,None]).abs()-(base[:,None]+wrong-y[:,None]).abs()).detach()
        better=(gain_label[:,:1]>wrong_gain+0.01)&cv[:,:1]
        if bool(better.any()):
            # Only enforce correct pairing when hindsight actually says it is
            # more useful than a support/regime-local wrong pairing.
            pair=F.softplus(0.05-(gain_pred[:,:1]-wrong_pred))[better].mean()
    # Controller is trained on the proposed (ungated) residual, not its own
    # post-gate decision; avoids the trivial all-reject shortcut.
    proposal=out['ungated_correction'].float().detach()
    realized_gain=((base-y).abs()-(base+proposal-y).abs()).detach()
    pos=realized_gain.clamp_min(0); neg=(-realized_gain).clamp_min(0)
    gain_control=masked_mean(F.smooth_l1_loss(out['estimated_gain'].float(),pos,reduction='none'),valid)
    harm_control=masked_mean(F.smooth_l1_loss(out['estimated_harm'].float(),neg,reduction='none'),valid)
    harm_bce=F.binary_cross_entropy_with_logits(out['harm_logit'].float()[valid],
                                    (realized_gain[valid]<0).float())
    correction_harm=((out['prediction'].float()-y).abs()-(base-y).abs()).clamp_min(0)
    tail=valid_cvar(correction_harm,valid,float(cfg.get('v4',{}).get('tail_fraction',.1)))
    cfg4=cfg.get('v4',{})
    total=fc+float(cfg['training'].get('rmse_weight',.05))*rmse
    total=total+float(cfg4.get('rank_weight',.03))*ranking
    total=total+float(cfg4.get('utility_weight',.05))*utility
    total=total+float(cfg4.get('pair_weight',.005))*pair
    if stage!='transfer':
        control=gain_control+harm_control+0.25*harm_bce
        total=total+float(cfg4.get('control_weight',.03))*control+float(cfg4.get('tail_weight',.005))*tail
    else:control=zero
    values=dict(mae=fc,rmse=rmse,forecast=fc,ranking=ranking,utility=utility,
                pair=pair,control=control,tail=tail,
                actual_gain=masked_mean(realized_gain,valid),
                harm_rate=masked_mean((realized_gain<0).float(),valid),
                gate=out['gate'].float().mean(),
                correction_size=masked_mean(out['correction'].float().abs(),valid),total=total)
    return total,{k:v.detach() for k,v in values.items()}
