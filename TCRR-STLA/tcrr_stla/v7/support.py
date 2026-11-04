"""TRAIN-history-only query/source compatibility features.

All statistics use query-observed entries or source historical validity.
No query future or target from another variable is accessed.
"""
from __future__ import annotations
import torch

FEATURE_NAMES=(
    'query_coverage','source_coverage','overlap','intervention',
    'recent_overlap','recent_intervention','observed_agreement',
    'has_overlap','query_missing_age','query_longest_run',
    'query_neighbour_coverage','source_neighbour_coverage',
    'overlap_neighbour','query_recent_coverage',
)

def _neighbor_mean(value,neighbors,neighbor_valid=None):
    if neighbors is None or neighbors.numel()==0:return value
    if neighbors.shape[0]!=value.shape[-1]:raise ValueError('Functional neighbor node mismatch')
    ids=neighbors.to(value.device).long()
    if neighbor_valid is None:
        return value[...,ids].mean(-1)
    weights=neighbor_valid.to(value.device).float()
    if weights.shape!=ids.shape:raise ValueError('Functional neighbor validity mismatch')
    return (value[...,ids]*weights).sum(-1)/weights.sum(-1).clamp_min(1)

@torch.no_grad()
def pair_support_features(query_x,query_mask,source_x,source_valid,neighbors=None,neighbor_valid=None):
    """Return raw [B,K,N,14] reusable, nonlearned support features.

    query [B,H,N], source [B,K,H,N]. No values at hidden query
    locations are read. Zero-overlap agreement is defined to be zero.
    """
    B,H,N=query_mask.shape
    if query_x.shape!=(B,H,N):raise ValueError('Query shape mismatch')
    if source_x.ndim!=4 or source_x.shape!=source_valid.shape or source_x.shape[0]!=B or source_x.shape[2:]!=(H,N):
        raise ValueError('Source shape mismatch')
    K=source_x.shape[1]
    q=query_mask.float().clamp(0,1)[:,None]
    s=source_valid.float().clamp(0,1)
    overlap=q*s
    intervention=(1-q)*s
    count=overlap.sum(2)
    x=torch.nan_to_num(query_x.float())*query_mask.float()
    sx=torch.nan_to_num(source_x.float())*s
    agreement=((sx-x[:,None]).abs()*overlap).sum(2)/count.clamp_min(1)
    agreement=torch.exp(-agreement.clamp(0,15))
    agreement=torch.where(count>0,agreement,torch.zeros_like(agreement))
    qc=q.mean(2).expand(B,K,N)
    sc=s.mean(2)
    oc=overlap.mean(2)
    ic=intervention.mean(2)
    rec=max(1,H//4)
    recent_overlap=overlap[:,:,-rec:].mean(2)
    recent_intervention=intervention[:,:,-rec:].mean(2)
    recent_coverage=q[:,:,-rec:].mean(2).expand(B,K,N)
    missing=(1-q[:,0]);run=torch.zeros((B,N),device=query_x.device,dtype=torch.float32);longest=run.clone()
    for t in range(H):
        run=(run+1)*missing[:,t]
        longest=torch.maximum(longest,run)
    age=(run/H)[:,None].expand(B,K,N)
    longest=(longest/H)[:,None].expand(B,K,N)
    raw=torch.stack((qc,sc,oc,ic,recent_overlap,recent_intervention,agreement,
                     (count>0).float(),age,longest,
                     _neighbor_mean(qc,neighbors,neighbor_valid),_neighbor_mean(sc,neighbors,neighbor_valid),
                     _neighbor_mean(oc,neighbors,neighbor_valid),recent_coverage),-1)
    return torch.nan_to_num(raw,nan=0.,posinf=0.,neginf=0.).clamp(-1e5,1e5)


def mask_temporal_features(mask):
    """[B,N,5]: coverage, recent coverage, final missing age,
    longest gap, longest recent run. Valid for completely missing history.
    """
    if mask.ndim!=3:raise ValueError('Expected [B,H,N] mask')
    B,H,N=mask.shape
    m=(mask>0.5).float()
    miss=1-m
    age=torch.zeros((B,N),device=m.device,dtype=m.dtype)
    longest=age.clone()
    for t in range(H):
        age=(age+1)*miss[:,t]
        longest=torch.maximum(longest,age)
    r=max(1,H//4)
    return torch.stack((m.mean(1),m[:,-r:].mean(1),age/H,longest/H,miss[:,-r:].mean(1)),dim=-1)
