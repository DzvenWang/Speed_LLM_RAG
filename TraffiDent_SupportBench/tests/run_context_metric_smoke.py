#!/usr/bin/env python3
"""Smoke-test v1.2.3 predictive-context metric semantics."""
from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supportbench.dependency import dependency_context_loss, summarize_dependency_context_loss
from supportbench.graph import predictive_source_contribution

# Three targets. Target 2 has no predictive neighbours so Chapter-3 network
# mean must still divide by N=3 rather than only the two non-empty targets.
dep_ids = np.asarray([
    [1, 2],
    [0, -1],
    [-1, -1],
], dtype=np.int64)
dep_weights = np.asarray([
    [0.75, 0.25],
    [1.00, 0.00],
    [0.00, 0.00],
], dtype=np.float32)

# Permanent full-history outage of source node 0.
mask = np.zeros((12, 3), dtype=bool)
mask[:, 0] = True
loss = dependency_context_loss(mask, dep_ids, dep_weights)
summary = summarize_dependency_context_loss(loss, quantile=0.75)

assert np.allclose(loss, [0.0, 1.0, 0.0]), loss
assert np.isclose(summary['network_mean_context_loss'], 1.0 / 3.0), summary
assert np.isclose(summary['context_loss_quantile_value'], 0.5), summary
assert not np.isclose(summary['network_mean_context_loss'], summary['context_loss_quantile_value'])

# The predictive-layout marginal objective must equal the formal network mean.
contribution = predictive_source_contribution(dep_ids, dep_weights, n_nodes=3)
assert np.isclose(contribution[0], 1.0 / 3.0), contribution
assert np.isclose(contribution[[0]].sum(), summary['network_mean_context_loss']), (contribution, summary)

print('CONTEXT METRIC SMOKE PASS')
print('per_target=', loss.tolist())
print('network_mean=', summary['network_mean_context_loss'])
print('legacy_q75=', summary['context_loss_quantile_value'])
