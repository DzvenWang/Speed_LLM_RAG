#!/usr/bin/env python3
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.compat.lrtc_tmcp import recover_batch


def make_case():
    torch.manual_seed(10)
    x = torch.rand(3, 5, 4, 12) * 60.0
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[0, 1:4, -1, 3:8] = False
    mask[1, 0:2, -1, 0:6] = False
    mask[2, 3:5, -1, 6:12] = False
    sparse = torch.where(mask, x, torch.zeros_like(x))
    return x, mask, sparse


x, mask, sparse = make_case()

# 1) Multi-day contract and immutable observed values.
y, info = recover_batch(
    sparse, mask, max_iter=3, decomposition='gram_eigh', active_convergence=True
)
assert y.shape == x.shape
assert torch.isfinite(y).all()
assert torch.equal(y[mask], x[mask]), 'observed entries must be protected exactly'
assert info.iterations >= 1
assert info.decomposition == 'gram_eigh'

# 2) Exact-decomposition contract: Gram-eigh must reproduce direct SVD for the
# same fixed number of TMCP iterations (active stopping disabled here).
y_svd, _ = recover_batch(
    sparse, mask, max_iter=3, decomposition='direct_svd', active_convergence=False
)
y_gram, _ = recover_batch(
    sparse, mask, max_iter=3, decomposition='gram_eigh', active_convergence=False
)
max_abs = float((y_svd - y_gram).abs().max().item())
assert torch.allclose(y_svd, y_gram, rtol=2e-4, atol=2e-4), max_abs

# 3) Active convergence must match running each query independently with the
# same stop rule. A loose tolerance makes queries terminate at differing times.
y_active, active_info = recover_batch(
    sparse, mask, max_iter=12, tol=2e-2,
    decomposition='gram_eigh', active_convergence=True,
)
parts = []
for i in range(sparse.shape[0]):
    yi, _ = recover_batch(
        sparse[i:i+1], mask[i:i+1], max_iter=12, tol=2e-2,
        decomposition='gram_eigh', active_convergence=True,
    )
    parts.append(yi)
y_individual = torch.cat(parts, dim=0)
active_max_abs = float((y_active - y_individual).abs().max().item())
assert torch.allclose(y_active, y_individual, rtol=2e-4, atol=2e-4), active_max_abs

print('LRTC-TMCP CONTRACT SMOKE PASS', {
    'shape': tuple(y.shape),
    'gram_vs_direct_max_abs': max_abs,
    'active_vs_individual_max_abs': active_max_abs,
    'active_info': active_info,
})
