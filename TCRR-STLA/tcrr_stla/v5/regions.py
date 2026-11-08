"""Deterministic groups from *train-only functional* neighbours, not road topology.

Group membership is invariant across batches and uses no query future labels.
Singletons are used where no functional neighbours are available.
"""
import torch


def functional_region_ids(neighbours, nodes, max_group_size=12, device=None):
    if nodes < 1 or max_group_size < 1:
        raise ValueError('Invalid functional-region dimensions')
    if neighbours is None or neighbours.numel() == 0:
        return torch.arange(nodes, dtype=torch.long, device=device)
    if neighbours.ndim != 2 or neighbours.shape[0] != nodes:
        raise ValueError('Neighbours must have shape [N,L]')
    rows=neighbours.detach().cpu().long().tolist()
    assigned=[-1]*nodes
    group=0
    for seed in range(nodes):
        if assigned[seed] >= 0:
            continue
        assigned[seed]=group
        frontier=[seed]
        size=1
        while frontier and size<max_group_size:
            node=frontier.pop(0)
            for candidate in rows[node]:
                if not (0<=candidate<nodes) or assigned[candidate]>=0:
                    continue
                assigned[candidate]=group
                frontier.append(candidate)
                size+=1
                if size>=max_group_size:
                    break
        group+=1
    return torch.tensor(assigned,dtype=torch.long,device=device)


def region_mean(x,region_ids):
    """[B,P,N,C] -> same shape; every group has exactly shared coarse mean."""
    B,P,N,C=x.shape
    if region_ids.shape != (N,):
        raise ValueError('Invalid region assignment')
    R=int(region_ids.max().item())+1
    ids=region_ids.view(1,1,N,1).expand(B,P,N,C)
    sums=x.new_zeros(B,P,R,C)
    sums.scatter_add_(2,ids,x)
    counts=x.new_zeros(B,P,R,1)
    counts.scatter_add_(2,region_ids.view(1,1,N,1).expand(B,P,N,1),x.new_ones(B,P,N,1))
    return (sums / counts.clamp_min(1)).gather(2,ids)
