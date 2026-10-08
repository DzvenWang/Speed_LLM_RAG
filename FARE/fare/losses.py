from __future__ import annotations
import torch
import torch.nn.functional as F


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.bool()
    if not torch.any(m):
        return (pred - target).abs().mean() * 0.0
    return (pred - target).abs()[m].mean()


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.bool()
    if not torch.any(m):
        return ((pred - target) ** 2).mean() * 0.0
    return ((pred - target) ** 2)[m].mean()


def per_query_masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.bool()
    diff = (pred - target).abs() * m.float()
    num = diff.flatten(1).sum(1)
    den = m.flatten(1).sum(1).clamp_min(1)
    return num / den


def baseline_relative_no_harm(
    expert_pred: torch.Tensor,
    baseline_pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    e = per_query_masked_l1(expert_pred, target, mask)
    b = per_query_masked_l1(baseline_pred, target, mask).detach()
    penalty = torch.relu(e - b + float(margin)).mean()
    return penalty, e.mean(), b.mean()


def pairwise_utility_ranking_loss(
    scores: torch.Tensor,
    utilities: torch.Tensor,
    valid: torch.Tensor | None = None,
    min_gap: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pairwise ordering loss invariant to absolute utility magnitude."""
    if scores.ndim != 2 or utilities.shape != scores.shape:
        raise ValueError("scores/utilities must be [B,K]")
    if valid is None:
        valid = torch.isfinite(utilities)
    else:
        valid = valid.bool() & torch.isfinite(utilities)
    losses=[]; correct=[]; counts=[]
    K=scores.shape[1]
    tri=torch.triu(torch.ones(K,K,dtype=torch.bool,device=scores.device),diagonal=1)
    for q in range(scores.shape[0]):
        vm=valid[q]
        pair_valid=(vm[:,None] & vm[None,:] & tri)
        du=utilities[q,:,None]-utilities[q,None,:]
        pair_valid=pair_valid & (du.abs() >= float(min_gap))
        if not torch.any(pair_valid):
            continue
        ds=scores[q,:,None]-scores[q,None,:]
        direction=torch.sign(du[pair_valid]).detach()
        margin=ds[pair_valid]*direction
        losses.append(F.softplus(-margin).mean())
        correct.append((margin>0).float().mean())
        counts.append(pair_valid.sum().float())
    if not losses:
        z=scores.sum()*0.0
        return z,z,torch.zeros((),device=scores.device)
    return torch.stack(losses).mean(),torch.stack(correct).mean(),torch.stack(counts).sum()


def query_gate_mean(gate: torch.Tensor, artificial: torch.Tensor) -> torch.Tensor:
    m=artificial.bool().float()
    num=(gate*m).flatten(1).sum(1)
    den=m.flatten(1).sum(1).clamp_min(1.0)
    return num/den


def restoration_reliability_target(estimate: torch.Tensor, full: torch.Tensor,
                                   artificial: torch.Tensor) -> torch.Tensor:
    """Cell-level local trust target from supervised restoration correctness.

    Query-level forecast utility is learned separately by the intervention
    landscape.  Reliability only answers *where* a raw estimate is locally
    trustworthy.  Per-query scaling avoids favoring easy queries globally.
    """
    m=artificial.bool().float()
    err=(estimate-full).abs()
    den=m.flatten(1).sum(1).clamp_min(1.0)
    scale=(err*m).flatten(1).sum(1)/den
    target=torch.exp(-err/scale[:,None,None].clamp_min(1e-4))
    return target.detach()


def reliability_supervision_loss(reliability: torch.Tensor, estimate: torch.Tensor,
                                 full: torch.Tensor, artificial: torch.Tensor):
    target=restoration_reliability_target(estimate,full,artificial)
    m=artificial.bool()
    if not torch.any(m):
        z=reliability.sum()*0.0
        return z,z,target
    loss=F.smooth_l1_loss(reliability[m],target[m])
    mae=(reliability[m]-target[m]).abs().mean()
    return loss,mae,target


def intervention_landscape_loss(predicted_scaled: torch.Tensor,
                                target_utility: torch.Tensor,
                                alpha_grid: torch.Tensor,
                                utility_scale: float = 100.0,
                                pairwise_min_gap: float = 1e-4,
                                choice_temperature: float = 1.0):
    """Supervise the complete query-level intervention-response landscape.

    predicted_scaled is trained in percentage-point utility units.  Targets are
    stored as relative utility fractions and scaled here for numerical stability.
    Returns regression, choice CE, pairwise loss/accuracy and alpha diagnostics.
    """
    valid=torch.isfinite(target_utility)
    if target_utility.ndim != 2 or predicted_scaled.shape != target_utility.shape:
        raise ValueError("landscape tensors must be [B,A]")
    if not torch.all(valid):
        # Landscape generation is all-or-nothing per query; fail loudly if an
        # incomplete curve reaches training.
        if torch.any(valid.sum(1) != target_utility.shape[1]):
            raise ValueError("Incomplete intervention utility curve in batch")
    target_scaled=torch.nan_to_num(target_utility,nan=0.0)*float(utility_scale)
    reg=F.smooth_l1_loss(predicted_scaled,target_scaled)
    best_idx=torch.argmax(target_scaled,dim=1)
    ce=F.cross_entropy(predicted_scaled/max(float(choice_temperature),1e-6),best_idx)
    pair,pair_acc,pairs=pairwise_utility_ranking_loss(
        predicted_scaled,target_scaled,torch.isfinite(target_scaled),float(pairwise_min_gap)*float(utility_scale)
    )
    pred_idx=torch.argmax(predicted_scaled,dim=1)
    alpha_grid=alpha_grid.to(predicted_scaled.device)
    pred_alpha=alpha_grid[pred_idx]
    target_alpha=alpha_grid[best_idx]
    alpha_acc=(pred_idx==best_idx).float().mean()
    alpha_mae=(pred_alpha-target_alpha).abs().mean()
    curve_mae=(predicted_scaled-target_scaled).abs().mean()
    positive=(pred_alpha>0).float().mean()
    oracle_positive=(target_alpha>0).float().mean()
    return {
        'regression':reg,'choice':ce,'pairwise':pair,'pair_acc':pair_acc,'pairs':pairs,
        'curve_mae_pp':curve_mae,'alpha_acc':alpha_acc,'alpha_mae':alpha_mae,
        'pred_alpha_mean':pred_alpha.mean(),'oracle_alpha_mean':target_alpha.mean(),
        'pred_positive_rate':positive,'oracle_positive_rate':oracle_positive,
        'target_alpha':target_alpha,'target_scaled':target_scaled,
    }
