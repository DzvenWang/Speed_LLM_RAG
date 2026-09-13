#!/usr/bin/env python3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
files=[
    ROOT/'scripts/prepare_data.py', ROOT/'scripts/validate_cache.py', ROOT/'scripts/run_experiment.py',
    ROOT/'scripts/summarize_screening.py', ROOT/'scripts/summarize_formal.py', ROOT/'scripts/evaluate_frozen_test.py',
    ROOT/'scripts/refresh_screening_metrics.py', ROOT/'scripts/smoke_recent_gpu.py',
    ROOT/'STDN_Special/audit_time_features.py', ROOT/'STDN_Special/inspect_cache.py', ROOT/'STDN_Special/run_diagnostic.py',
]
for f in files:
    t=f.read_text()
    if "'_cache'" in t or '"_cache"' in t or 'ensure_consumer_alias' in t:
        raise SystemExit(f'FAIL local cache alias reference remains in {f}')
helper=(ROOT/'backbone/shared_cache.py').read_text()
assert 'experiment_common.cache' in helper and 'cache_dir' in helper
print('BACKBONE DIRECT EXPERIMENTCACHE CONTRACT SMOKE PASS')
