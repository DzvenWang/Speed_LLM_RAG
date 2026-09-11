#!/usr/bin/env python3
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared_protocol.sampler import all_unique_group_pick, balanced_unique_group_pick

fams = [
    'temporal_outage',
    'regional_outage',
    'directional_outage',
    'predictive_regional_outage',
    'predictive_directional_outage',
]

# Development-style exact balanced selection.
rows = []
for i in range(200):
    for f in fams:
        if not (i % 11 == 0 and f == 'predictive_directional_outage'):
            rows.append({
                'view_id': f'v{i}_{f}',
                'view_group_id': f'g{i}',
                'window_start': i,
                'family': f,
            })
df = pd.DataFrame(rows)
out = balanced_unique_group_pick(df, fams, 100, 10)
assert len(out) == 100 and out.view_group_id.nunique() == 100
assert set(out.family.value_counts()) == {20}

# Formal-style full-train selection: keep every group even when exact family
# quotas are structurally impossible.
rows = []
for i in range(50):
    available = (
        ['temporal_outage'] if i < 20 else
        ['regional_outage'] if i < 30 else
        ['directional_outage', 'predictive_regional_outage'] if i < 40 else
        ['predictive_regional_outage', 'predictive_directional_outage']
    )
    for f in available:
        rows.append({
            'view_id': f'formal_v{i}_{f}',
            'view_group_id': f'formal_g{i}',
            'window_start': i,
            'family': f,
        })
formal_df = pd.DataFrame(rows)
formal = all_unique_group_pick(formal_df, fams, 20)
assert len(formal) == 50
assert formal.view_group_id.nunique() == 50
assert set(formal.view_group_id) == {f'formal_g{i}' for i in range(50)}
assert set(formal.family).issubset(set(fams))
# Determinism check.
formal2 = all_unique_group_pick(formal_df, fams, 20)
pd.testing.assert_frame_equal(formal.reset_index(drop=True), formal2.reset_index(drop=True))

print('SHARED PROTOCOL SAMPLER SMOKE PASS')
