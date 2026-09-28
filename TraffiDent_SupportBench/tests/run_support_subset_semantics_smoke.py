#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supportbench.sensing import _temporal_outage_shape
from supportbench.dependency import dependency_context_loss

H, N = 12, 388
threshold = 0.40
k, duration = _temporal_outage_shape(0.20, H, N, 0.75)
assert (k, duration) == (291, 3), (k, duration)

# Any source node is absent for at most 3/12 of history; dependency_context_loss
# is a convex combination of source missing fractions, so it cannot reach 0.40.
mask = np.zeros((H, N), dtype=bool)
mask[:duration, :k] = True
ids = np.tile(np.arange(8, dtype=np.int64), (N, 1))
weights = np.full((N, 8), 1/8, dtype=np.float32)
loss = dependency_context_loss(mask, ids, weights)
assert float(loss.max()) <= duration / H + 1e-7
assert duration / H < threshold

# At s40 the current construction has a 6/12 per-source upper bound, so the
# threshold becomes structurally attainable (not necessarily met by every query).
_, duration40 = _temporal_outage_shape(0.40, H, N, 0.75)
assert duration40 / H >= threshold
print('SUPPORT SUBSET SEMANTICS SMOKE PASS')
