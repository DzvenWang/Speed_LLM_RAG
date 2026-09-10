#!/usr/bin/env python3
# Static contract smoke: Baseline v0.4.0 must consume Shared Protocol, never own sampling.
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'baseline/data.py').read_text(); assert 'balanced_unique_group_pick' not in s and 'load_protocol_views' in s
p=(root/'scripts/prepare_data.py').read_text(); assert 'protocol_stage_hash' in p and 'data_config_sha256' in p
print('BASELINE SHARED PROTOCOL CONSUMER SMOKE PASS')
