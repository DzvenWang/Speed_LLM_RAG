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
    ids = retrieved['ids']
    nodes = regions.region_node_ids  # [G,S], padded
    node_ok = regions.region_node_valid
    with torch.no_grad():
        hist, mem_ok = memory.fetch_nodes(ids,nodes)  # [B,G,K,H,S]
        expected = xfull[:,:,nodes].permute(0,2,1,3)[:, :, None]
        wanted = target[:,:,nodes].permute(0,2,1,3)[:, :, None]
        valid = wanted & mem_ok & node_ok[None,:,None,None,:]
        count = valid.sum((-1,-2))
        error = ((hist-expected).abs()*valid).sum((-1,-2))/count.clamp_min(1)
        eligible = (count>0) & retrieved['valid']
    return error,eligible


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


def forecast_utility_ranking_loss(batch, retrieved, memory, regions, temperature=0.35):
    """Train-only forecast-utility supervision using legal historical futures.

    Scores are compared within each group's selected candidates; candidate
    historical futures are aligned by observed current-region mean.  Labels
    never enter the model forward or the inference retrieval feature vector.
    """
    weights=retrieved['weights']
    if memory.future is None:
        return weights.sum()*0, {'utility_eligible_fraction':weights.new_zeros(())}
    with torch.no_grad():
        ids=retrieved['ids']
        nodes=regions.region_node_ids
        node_ok=regions.region_node_valid
        b,g,k=ids.shape
        target=batch['y_norm'].float()   # [B,P,N]
        mask=batch['future_valid'].bool()
        x=batch['x_obs'].float()
        observed=batch['obs_mask'].bool()
        # Only last-step history values and required region future nodes are read.
        past,past_ok=memory.fetch_nodes(ids,nodes,time_index=-1) # [B,G,K,S]
        cur=x[:,-1,nodes]        # [B,G,S]
        obs=observed[:,-1,nodes] # [B,G,S]
        common=past_ok & obs[:,:,None,:] & node_ok[None,:,None,:]
        common_count=common.sum(-1)
        shift=((cur[:,:,None,:]-past)*common).sum(-1)/common_count.clamp_min(1)
        future,future_ok=memory.fetch_nodes(ids,nodes,kind='future') # [B,G,K,P,S]
        y=target[:,:,nodes].permute(0,2,1,3)[:, :, None]
        ym=mask[:,:,nodes].permute(0,2,1,3)[:, :, None]
        valid=future_ok & ym & node_ok[None,:,None,None,:]
        valid_count=valid.sum((-1,-2))
        error=((future+shift[...,None,None]-y).abs()*valid).sum((-1,-2))/valid_count.clamp_min(1)
        # Keep the exact persistence-reference denominator used in v2.0.0.
        # A candidate axis is intentionally not present in the reference.
        ref_y=target[:,:,nodes].permute(0,2,1,3)
        ref_m=mask[:,:,nodes].permute(0,2,1,3) & node_ok[None,:,None,:]
        reference=((cur[:,:,None,:]-ref_y).abs()*ref_m).sum((-1,-2))/ref_m.sum((-1,-2)).clamp_min(1)
        utility=reference[:,:,None]-error
        eligible=(valid_count>0)&(common_count>0)&retrieved['valid']
        teacher=torch.softmax((utility/temperature).masked_fill(~eligible,-1e4),-1)*eligible.float()
        teacher=teacher/teacher.sum(-1,keepdim=True).clamp_min(1e-9)
        ok=(eligible.sum(-1)>=2)
    q=weights*eligible.float()
    q=q/q.sum(-1,keepdim=True).clamp_min(1e-9)
    ce=-(teacher*q.clamp_min(1e-9).log()).sum(-1)
    loss=(ce*ok).sum()/ok.sum().clamp_min(1)
    # All-ineligible groups have zero CE by construction.
    positive=(utility>0)&eligible
    return loss,{'utility_eligible_fraction':ok.float().mean().detach(),
                 'utility_positive_fraction':positive.float().sum().div(eligible.float().sum().clamp_min(1)).detach()}
