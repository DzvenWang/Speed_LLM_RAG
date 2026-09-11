from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.runners.common import _valid_datetime_candidate, as_unicode_array

# Prediction IDs must be pickle-free.
a = as_unicode_array(['v1', 'longer-view-id'])
assert a.dtype.kind == 'U'

# A numeric row index must NOT be accepted as a datetime axis.
assert _valid_datetime_candidate(pd.Series(np.arange(1000)), 100) is None

# A genuine 5-minute timestamp column must be accepted.
ts = pd.Series(pd.date_range('2024-01-01', periods=1000, freq='5min').astype(str))
parsed = _valid_datetime_candidate(ts, 100)
assert parsed is not None and len(parsed) == 100
assert (parsed[1] - parsed[0]).total_seconds() == 300

raft = (ROOT / 'baseline' / 'runners' / 'raft_runner.py').read_text()
assert "tr.detach().to(device" in raft and "va.detach().to(device" in raft
saraf = (ROOT / 'baseline' / 'runners' / 'saraf_runner.py').read_text()
assert 'SARAF TIME BONUS PROBE' in saraf and 'effectively constant' in saraf
print('BASELINE 20260911 HOTFIX SMOKE PASS')
