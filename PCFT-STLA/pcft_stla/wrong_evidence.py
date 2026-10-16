"""Deterministic, non-overlapping Train-only negative cases; batch invariant."""
from __future__ import annotations
import numpy as np
import torch

def legal_wrong_indices(query_starts, candidates, train_starts, span, device):
    starts=np.asarray(train_starts,dtype=np.int64)
    q=np.asarray(query_starts.detach().cpu().numpy() if torch.is_tensor(query_starts) else query_starts,dtype=np.int64)
    if not len(starts):raise RuntimeError('No Train historical memory for negative evidence')
    ans=np.empty((len(q),int(candidates)),dtype=np.int64)
    for b,window in enumerate(q):
        for k in range(int(candidates)):
            # Stable per-query key; does not depend on evaluation batch grouping.
            v=(int(window)*1103515245+(k+1)*2654435761+0x9e3779b9) % len(starts)
            for tries in range(len(starts)):
                if abs(int(starts[v])-int(window))>=span:break
                v=(v+1)%len(starts)
            else:raise RuntimeError('No temporally legal Train evidence for negative control')
            ans[b,k]=v
    return torch.as_tensor(ans,device=device,dtype=torch.long)
