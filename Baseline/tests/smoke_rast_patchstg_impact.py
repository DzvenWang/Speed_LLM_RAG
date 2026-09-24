#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.runners.retrieval_impact import (  # noqa: E402
    IMPACT_KEYS,
    accumulate_impact,
    empty_impact_stats,
    finalize_impact,
)

# Three valid targets: one helpful, one harmful, one neutral. Fourth is masked out.
base = torch.tensor([[[2.0, 2.0, 2.0, 99.0]]], dtype=torch.float32)
final = torch.tensor([[[1.0, 3.0, 2.0, -99.0]]], dtype=torch.float32)
y = torch.tensor([[[0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
valid = torch.tensor([[[1, 1, 1, 0]]], dtype=torch.bool)

s = empty_impact_stats()
accumulate_impact(s, final, base, y, valid)
out = finalize_impact(s)
assert out is not None
assert tuple(out.keys()) == IMPACT_KEYS
assert out['count'] == 3
assert math.isclose(out['helpful_fraction'], 1 / 3, rel_tol=0, abs_tol=1e-12)
assert math.isclose(out['harmful_fraction'], 1 / 3, rel_tol=0, abs_tol=1e-12)
assert math.isclose(out['neutral_fraction'], 1 / 3, rel_tol=0, abs_tol=1e-12)
assert math.isclose(out['correction_abs_mean_norm'], 2 / 3, rel_tol=0, abs_tol=1e-12)
assert math.isclose(out['correction_rms_norm'], math.sqrt(2 / 3), rel_tol=0, abs_tol=1e-12)
assert math.isclose(out['correction_max_abs_norm'], 1.0, rel_tol=0, abs_tol=1e-12)
assert math.isclose(out['base_MAE_norm'], 2.0, rel_tol=0, abs_tol=1e-12)
assert math.isclose(out['final_MAE_norm'], 2.0, rel_tol=0, abs_tol=1e-12)
assert math.isclose(out['mean_abs_error_change_norm'], 0.0, rel_tol=0, abs_tol=1e-12)

s2 = empty_impact_stats()
accumulate_impact(s2, final, base, y, torch.zeros_like(valid))
assert finalize_impact(s2) is None

print('RAST-PATCHSTG IMPACT SMOKE PASS', out)
