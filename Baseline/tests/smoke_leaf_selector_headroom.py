#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("BASELINE_ROOT", str(next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd()))))
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.analysis.leaf_selector_headroom import (
    apply_query_node_choices,
    build_query_node_oracle,
    choice_statistics,
    relative_gain,
)

# B=2, P=3, N=2.  Each query-node must choose one branch for all horizons.
y = np.zeros((2, 3, 2), dtype=np.float32)
valid = np.ones_like(y, dtype=np.uint8)

# q0n0 Graph wins; q0n1 Hyper wins; q1n0 tie -> Hyper; q1n1 Graph wins.
g = np.array(
    [
        [[1.0, 4.0], [1.0, 4.0], [1.0, 4.0]],
        [[2.0, 1.0], [2.0, 1.0], [2.0, 1.0]],
    ],
    dtype=np.float32,
)
h = np.array(
    [
        [[3.0, 1.0], [3.0, 1.0], [3.0, 1.0]],
        [[2.0, 5.0], [2.0, 5.0], [2.0, 5.0]],
    ],
    dtype=np.float32,
)

oracle = build_query_node_oracle(g, h, y, valid, tie_break="hyper")
expected = np.asarray([[1, 2], [2, 1]], dtype=np.uint8)
assert np.array_equal(oracle.choices, expected), (oracle.choices, expected)

pred = apply_query_node_choices(g, h, oracle.choices)
assert pred.shape == y.shape
assert np.all(pred[0, :, 0] == g[0, :, 0])
assert np.all(pred[0, :, 1] == h[0, :, 1])
assert np.all(pred[1, :, 0] == h[1, :, 0])
assert np.all(pred[1, :, 1] == g[1, :, 1])

stats = choice_statistics(oracle)
assert stats["eligible_query_nodes"] == 4
assert abs(stats["graph_strict_win_fraction"] - 0.5) < 1e-12
assert abs(stats["hyper_strict_win_fraction"] - 0.25) < 1e-12
assert abs(stats["tie_fraction"] - 0.25) < 1e-12
assert abs(relative_gain(2.0, 1.5) - 0.25) < 1e-12

# No-valid pairs must be excluded from win rates and fall back deterministically.
valid2 = valid.copy(); valid2[1, :, 1] = 0
oracle2 = build_query_node_oracle(g, h, y, valid2, tie_break="hyper")
assert not oracle2.eligible[1, 1]
assert oracle2.choices[1, 1] == 2
assert choice_statistics(oracle2)["eligible_query_nodes"] == 3

print("LEAF SELECTOR HEADROOM SMOKE PASS")
