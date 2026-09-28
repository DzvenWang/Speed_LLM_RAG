#!/usr/bin/env python3
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline.lrtc_causal_history import LRTCCanonicalHistory

H = 12
N = 5
DAY = 288
DAYS = 4
T = DAY * 20
speed = (np.arange(T * N, dtype=np.float32).reshape(T, N) % 80.0) + 1.0
valid = np.ones((T, N), dtype=np.uint8)
source = LRTCCanonicalHistory(
    target_array=speed,
    target_channel=None,
    valid_array=valid,
    valid_channel=None,
    target_path=Path('synthetic_target.npy'),
    valid_path=Path('synthetic_valid.npy'),
    history=H,
    nodes=N,
    historical_days=DAYS,
    day_steps=DAY,
    observed_match_mae=0.0,
    raw_valid_agreement=1.0,
)

st = DAY * 10 + 30
current = speed[st:st + H].copy()
obs = np.ones((H, N), dtype=bool)
obs[2:8, 1:4] = False
v1, m1, starts1 = source.build_tensor(st, current, obs)
assert v1.shape == (N, DAYS + 1, H)
assert m1.shape == v1.shape
assert starts1[-1] == st
assert all(s + H <= st for s in starts1[:-1])
assert np.array_equal(m1[:, -1, :].T, obs)
assert np.all(v1[:, -1, :].T[~obs] == 0.0)
assert np.array_equal(v1[:, -1, :].T[obs], current[obs])

# Functional future-isolation check: mutate every value after the current H12.
speed2 = speed.copy()
speed2[st + H:] = 999999.0
source2 = LRTCCanonicalHistory(
    target_array=speed2,
    target_channel=None,
    valid_array=valid,
    valid_channel=None,
    target_path=Path('synthetic_target_mutated_future.npy'),
    valid_path=Path('synthetic_valid.npy'),
    history=H,
    nodes=N,
    historical_days=DAYS,
    day_steps=DAY,
    observed_match_mae=0.0,
    raw_valid_agreement=1.0,
)
v2, m2, starts2 = source2.build_tensor(st, current, obs)
np.testing.assert_array_equal(v1, v2)
np.testing.assert_array_equal(m1, m2)
assert starts1 == starts2

print('LRTC-TMCP CAUSAL HISTORY SMOKE PASS', {
    'tensor_shape': v1.shape,
    'historical_days': DAYS,
    'future_mutation_invariant': True,
    'current_hidden_preserved': True,
})
