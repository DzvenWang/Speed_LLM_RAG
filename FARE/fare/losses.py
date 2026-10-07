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
    """Pairwise utility ranking that is invariant to the absolute utility scale.

    For every informative candidate pair, the higher-utility support must receive
    the higher model score. This avoids the near-uniform softmax targets that
    occur when downstream utilities are only a few 1e-3 apart.

    Returns (loss, pair_accuracy, informative_pair_count).
    Random-order logistic baseline is ln(2) ~= 0.6931 and accuracy 0.5.
    """
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


def alpha_supervision_loss(gate: torch.Tensor, artificial: torch.Tensor, target_alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred=query_gate_mean(gate,artificial)
    target=target_alpha.float().reshape_as(pred)
    return F.smooth_l1_loss(pred,target), (pred-target).abs().mean()
