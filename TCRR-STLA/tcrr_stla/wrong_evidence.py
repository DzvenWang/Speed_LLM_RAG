"""Batch-independent legal Train-case mismatch for retrieval attribution."""
from __future__ import annotations
import numpy as np
import torch

def legal_wrong_indices(query_starts, candidates, train_starts, span, device):
    starts=np.asarray(train_starts,dtype=np.int64)
    q=np.asarray(query_starts.detach().cpu().numpy() if torch.is_tensor(query_starts) else query_starts,dtype=np.int64)
    if not len(starts):raise RuntimeError('Empty Train Memory')
    result=np.empty((len(q),candidates),dtype=np.int64)
    for b,start in enumerate(q):
        for k in range(candidates):
            j=(int(start)*1103515245+(k+1)*2654435761+0x9e3779b9)%len(starts)
            for _ in range(len(starts)):
                if abs(int(starts[j])-int(start))>=span:break
                j=(j+1)%len(starts)
            else:raise RuntimeError('No nonoverlapping train memory for negative control')
            result[b,k]=j
    return torch.as_tensor(result,device=device,dtype=torch.long)
