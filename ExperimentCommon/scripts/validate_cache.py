#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
for p in [ROOT, REPO_ROOT / 'Protocol', REPO_ROOT / 'TraffiDent_SupportBench']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiment_common.cache import validate_cache, cache_dir
from experiment_common.utils import load_yaml

p = argparse.ArgumentParser()
p.add_argument('--stage', choices=['development', 'formal', 'evaluation'], required=True)
p.add_argument('--config', default=str(ROOT / 'configs' / 'experiment_contract.yaml'))
p.add_argument('--validate-locked-evaluation', action='store_true', help='Validate a pre-materialized evaluation cache without granting model access.')
a = p.parse_args()
cfg = load_yaml(a.config)
ident = validate_cache(cfg, a.stage, allow_locked_evaluation_validation=a.validate_locked_evaluation)
print('PASS shared contract', ident['contract_version'])
print('PASS protocol', ident['protocol_name'], ident['protocol_version'])
print('PASS severity', f'{ident["severity"]:.2f}')
print('PASS Protocol stage hash', ident['protocol_stage_hash'])
print('PASS time feature fingerprint', ident['time_feature_fingerprint'])
print('CACHE_DIR=' + str(cache_dir(cfg, a.stage)))
print('SHARED CACHE VALIDATION PASS')
