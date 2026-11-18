"""Deterministic, temporally legal, unrelated Train rows for the wrong-evidence audit (vectorised)."""
from __future__ import annotations
import numpy as np
import torch


def legal_wrong_indices(query_starts, candidates, train_starts, span, device):
    starts = torch.as_tensor(np.asarray(train_starts, dtype=np.int64), device=device)
    q = torch.as_tensor(query_starts, device=device).long().flatten(); M = len(starts)
    if M == 0: raise RuntimeError('No Train memory for negative evidence')
    k = torch.arange(1, int(candidates) + 1, device=device, dtype=torch.long)
    # Stable per-(query, slot) hash: independent of batch composition.
    v = (q[:, None] * 1103515245 + k[None, :] * 2654435761 + 0x9e3779b9) % M
    for _ in range(M):
        bad = (starts[v] - q[:, None]).abs() < int(span)
        if not bool(bad.any()): return v
        v = torch.where(bad, (v + 1) % M, v)
    raise RuntimeError('No temporally legal Train evidence for negative control')
