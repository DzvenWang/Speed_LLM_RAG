"""Train-only paired historical futures for v2 candidate-level retrieval.

No test ground truth or test x_full is consumed by this module. Temporal exclusion
is performed while building candidate indices, before this module is called.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch


class FutureMemory:
    def __init__(self, formal_cache: str | Path, limit: int | None = None):
        root = Path(formal_cache)
        names = dict(x='train_x_full_norm.npy', y='train_y_norm.npy',
                     tod='train_tod_hist.npy', dow='train_dow_hist.npy')
        arrays = {key: np.load(root / filename, mmap_mode='r') for key, filename in names.items()}
        self.n = len(arrays['x']) if limit is None else min(limit, len(arrays['x']))
        if (root/'train_future_valid.npy').is_file():
            arrays['future_valid']=np.load(root/'train_future_valid.npy',mmap_mode='r')
        self.arrays = arrays
        self._gpu = {}

    def __len__(self):
        return self.n

    def preload(self, device):
        if torch.device(device).type != 'cuda':
            return False
        try:
            self._gpu = {k: torch.as_tensor(np.array(v[:self.n], copy=True), device=device)
                         for k, v in self.arrays.items()}
            return True
        except torch.cuda.OutOfMemoryError:
            self._gpu = {}
            torch.cuda.empty_cache()
            return False

    def fetch(self, indices: torch.Tensor, device: torch.device):
        if (indices < 0).any() or (indices >= self.n).any():
            raise IndexError('Retrieval indices must refer to the train-only memory')
        shape = tuple(indices.shape)
        out = {}
        if self._gpu:
            idx = indices.to(device=device, dtype=torch.long).reshape(-1)
            for key, val in self._gpu.items():
                raw = val[idx]
                out[key] = raw.reshape(*shape, *raw.shape[1:])
        else:
            idx = indices.detach().cpu().numpy().astype(np.int64).reshape(-1)
            for key, arr in self.arrays.items():
                val = torch.from_numpy(np.array(arr[idx], copy=True)).to(device)
                out[key] = val.reshape(*shape, *val.shape[1:])
        for key in ('x', 'y'):
            out[key] = torch.nan_to_num(out[key].float())
        for key in ('tod', 'dow'):
            out[key] = out[key].long()
        if 'future_valid' not in out:
            out['future_valid']=torch.ones_like(out['y'],dtype=torch.bool)
        else:
            out['future_valid']=out['future_valid'].bool()
        return out


def shortlist(cache, query_indices, per_period: int, device):
    """Preserve evidence across temporal resolutions instead of early averaging."""
    idx = np.asarray(query_indices, dtype=np.int64)
    if idx.size == 0 or int(idx.max()) >= cache.indices.shape[0]:
        raise IndexError('Retrieval cache is shorter than requested dataset split')
    k = min(int(per_period), cache.indices.shape[-1])
    ids = np.asarray(cache.indices[idx, :, :k], dtype=np.int64)
    scores = np.asarray(cache.scores[idx, :, :k], dtype=np.float32)
    coverage = np.repeat(np.asarray(cache.coverage[idx], dtype=np.float32)[:, :, None], k, axis=2)
    return (torch.as_tensor(ids.reshape(len(idx), -1), device=device),
            torch.as_tensor(scores.reshape(len(idx), -1), device=device),
            torch.as_tensor(coverage.reshape(len(idx), -1), device=device))


def counterfactual_residuals(stla, candidates, query_mask, chunk_size=32,
                             replay='matched'):
    """R=actual future - STLA forecast on each *train* candidate.

    Query mask is applied ONLY to candidate history. Candidate future is obtained
    from the train-only store. The STLA forward has no gradient, but parameters
    track any current fine-tuning state. Use replay='full' for ablation.
    """
    B, K, H, N = candidates['x'].shape
    if replay not in ('matched', 'full'):
        raise ValueError(replay)
    m = query_mask[:, None].expand(B, K, H, N)
    if replay == 'full':
        m = torch.ones_like(m)
    flat_x = (candidates['x'] * m).reshape(B*K, H, N)
    flat_m = m.reshape(B*K, H, N)
    flat_tod = candidates['tod'].reshape(B*K, H)
    flat_dow = candidates['dow'].reshape(B*K, H)
    pieces = []
    was_training = stla.training
    stla.eval()
    try:
        with torch.no_grad():
            for i in range(0, B*K, int(chunk_size)):
                p, _ = stla(flat_x[i:i+chunk_size], flat_m[i:i+chunk_size],
                            flat_tod[i:i+chunk_size], flat_dow[i:i+chunk_size])
                pieces.append(p.detach())
    finally:
        stla.train(was_training)
    pred = torch.cat(pieces, 0).reshape(B, K, *candidates['y'].shape[-2:])
    return candidates['y'] - pred
