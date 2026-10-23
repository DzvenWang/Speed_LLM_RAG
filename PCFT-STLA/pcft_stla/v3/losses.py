"""Train-label-only branch supervision for provenance propagation and action selection."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def weighted(x,m):
    return (x*m.float()).sum()/m.float().sum().clamp_min(1.)


def pcft_v3_losses(out,b,cfg):
    valid=b['future_valid'].bool(); y=b['y_norm'].float();zero=y.new_zeros(())
    base=out['base_prediction'].float().detach()
    level=out.get('v3_contribution','M0')
    branch=zero; utility=zero; tail=zero
    if level in ('M2','M3') and out.get('propagation_action') is not None:
        prop=out['propagation_action'].float();legacy=out['legacy_action'].float()
        avail=out.get('propagation_available',torch.ones_like(prop)).float()
        # Keep propagation branch learnable even if None currently dominates policy.
        mp=valid & (avail>1e-5)
        prop_loss=(base+prop-y).abs()
        base_loss=(base-y).abs()
        branch=weighted(prop_loss,mp)
        # Train branch against observed outcomes, without forcing a positive action.
        delta=(base_loss-prop_loss).detach()
        if level=='M3':
            gain=out['action_expected_gain'].float();harm=out['action_expected_harm'].float()
            dl=(base_loss-(base+legacy-y).abs()).detach()
            labels=torch.stack([dl,delta],-1)
            utility=weighted(F.smooth_l1_loss(gain,labels,reduction='none').mean(-1),valid)
            utility+=.25*weighted(F.smooth_l1_loss(harm,(-labels).clamp_min(0),reduction='none').mean(-1),valid)
            # Ranking against None uses actual action-level MAE gain labels.
            true_logits=torch.cat([labels,torch.zeros_like(labels[...,:1])],-1)/.25
            goal=torch.softmax(true_logits.detach(),-1)
            logits=out['action_logits'].float()
            utility+=.1*weighted(-(goal*torch.log_softmax(logits,dim=-1)).sum(-1),valid)
            err_final=(out['prediction'].float()-y).abs()
            tail=weighted((err_final-base_loss).clamp_min(0).square(),valid)
    v3=cfg.get('v3',{})
    total=float(v3.get('branch_weight',.025))*branch+float(v3.get('utility_weight',.03))*utility+float(v3.get('tail_weight',.01))*tail
    return total,{'v3_propagation_branch':branch.detach(), 'v3_action_utility':utility.detach(),
                  'v3_harm_tail':tail.detach()}
