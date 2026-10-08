"""Training-only candidate ranking targets and mask-aware listwise supervision.

Labels use artificially removed, originally valid training observations. They
must NEVER enter model.forward or inference retrieval features. Evaluation may
use the same labels for offline diagnostics only.
"""
from __future__ import annotations
import torch


def target_candidate_errors(batch, retrieved, memory, regions):
    """Errors for *legal retrieved candidates* on artificial target cells.

    Returns [B,G,K] normalized MAE and candidate eligibility. Each group owns a
    disjoint set of nodes; naturally invalid and observed cells are excluded.
    Computation is chunked over regions to avoid materializing B*G*K*H*N.
    """
    target = batch['artificial_mask'].bool() & batch['raw_valid'].bool()
    xfull = batch['x_full'].float()
    b,h,n = xfull.shape
    ids = retrieved['ids']; device = xfull.device
    n_groups, k = ids.shape[1], ids.shape[2]
    errors = xfull.new_zeros((b,n_groups,k))
    valid = torch.zeros((b,n_groups,k), dtype=torch.bool, device=device)
    for g in range(n_groups):
        node_ids = torch.where(regions.owner == g)[0]
        if not node_ids.numel(): continue
        # Memory indices [B,K], recovered historical values [B,K,H,S].
        hx = memory.x[ids[:,g]][:,:,:,node_ids]
        hv = memory.valid[ids[:,g]][:,:,:,node_ids].bool()
        wanted = target[:,:,node_ids][:,None,:,:] & hv
        count = wanted.sum((-1,-2))
        err = ((hx - xfull[:,:,node_ids][:,None,:,:]).abs() * wanted).sum((-1,-2))
        errors[:,g] = err / count.clamp_min(1)
        valid[:,g] = (count>0) & retrieved['valid'][:,g]
    return errors,valid


def target_recovery_ranking_loss(batch, retrieved, memory, regions,
                                 temperature=0.25):
    """Listwise ranking on teacher-quality historical restoration labels.

    Candidate IDs are selected using inference-only query features; labels
    differentiate top-K weights without leaking into retrieval at inference.
    No supervised target -> differentiable scalar zero, not NaN.
    """
    if temperature <= 0:raise ValueError('Ranking temperature must be positive')
    if 'x_full' not in batch or 'artificial_mask' not in batch:
        raise RuntimeError('Rank loss requires TRAIN/EVAL labels, never inference input')
    with torch.no_grad():
        err, valid = target_candidate_errors(batch,retrieved,memory,regions)
        # Among candidates with valid hidden target values, use smoother
        # teacher probabilities to avoid an unstable hard argmin.
        logits=(-err.float()/temperature).masked_fill(~valid,-1e4)
        teacher=torch.softmax(logits,dim=-1)*valid.float()
        teacher=teacher/teacher.sum(-1,keepdim=True).clamp_min(1e-8)
        eligible=valid.sum(-1)>=2
        best=err.masked_fill(~valid,float('inf')).min(-1).values
        best=torch.where(eligible,best,torch.zeros_like(best))
        count=eligible.sum().clamp_min(1)
        oracle_error=best.sum()/count
        uniform=(err*valid).sum(-1)/valid.sum(-1).clamp_min(1)
        uniform_error=(uniform*eligible).sum()/count
    weights=retrieved['weights']
    # Compare candidates on exactly the same *target-label-eligible* support.
    # Preserve the raw valid weight mass as a separate diagnostic.
    target_mass=(weights.detach()*valid.float()).sum(-1)
    comparable=weights*valid.float()
    comparable=comparable/comparable.sum(-1,keepdim=True).clamp_min(1e-8)
    ce=-(teacher*comparable.clamp_min(1e-8).log()).sum(-1)
    denom=eligible.sum().clamp_min(1)
    loss=(ce*eligible).sum()/denom
    if not eligible.any(): loss=weights.sum()*0
    # Weighted error for logging; not used to create gradient-through-labels.
    with torch.no_grad():
        rank_err=(err*comparable.detach()).sum(-1)
        rank_err=(rank_err*eligible).sum()/denom
        valid_mass=(target_mass*eligible).sum()/denom
    return loss,{'ranking_eligible_fraction':eligible.float().mean().detach(),
                 'ranking_selected_error':rank_err.detach(),
                 'ranking_oracle_error':oracle_error.detach(),
                 'ranking_uniform_error':uniform_error.detach(),
                 'ranking_target_valid_weight_mass':valid_mass.detach()}
