#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from backbone.adapters import canonicalize_bigst_prediction

B, P, N = 2, 12, 388
batch = {"y_norm": torch.zeros(B, P, N)}

# Official BigST native layout: [B, N, P].
x = torch.randn(B, N, P, requires_grad=True)
y = canonicalize_bigst_prediction(x, batch)
assert tuple(y.shape) == (B, P, N), tuple(y.shape)
y.sum().backward()
assert x.grad is not None

# Already-canonical output remains valid.
z = torch.randn(B, P, N)
z2 = canonicalize_bigst_prediction(z, batch)
assert tuple(z2.shape) == (B, P, N)

# Singleton output channel is supported.
w = torch.randn(B, N, P, 1)
w2 = canonicalize_bigst_prediction(w, batch)
assert tuple(w2.shape) == (B, P, N)

try:
    canonicalize_bigst_prediction(torch.randn(B, 7, 9), batch)
except RuntimeError:
    pass
else:
    raise AssertionError("invalid BigST shape should fail")

print("BIGST SHAPE CONTRACT SMOKE PASS")
