#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
for p in [ROOT, REPO_ROOT / 'Protocol', REPO_ROOT / 'TraffiDent_SupportBench']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiment_common import __version__
from experiment_common.cache import (
    load_protocol_stage_metadata, evaluation_access_granted, cache_dir,
)
from experiment_common.utils import load_yaml

p = argparse.ArgumentParser()
p.add_argument('--config', default=str(ROOT / 'configs' / 'experiment_contract.yaml'))
p.add_argument('--stage', choices=['development','formal','evaluation'])
a = p.parse_args()
cfg = load_yaml(a.config)
print('ExperimentCommon preflight v' + __version__)
print('config=', Path(a.config).resolve())
for key in ['root','cache_root','benchmark_root','protocol_artifact_root']:
    path = Path(cfg['project'][key])
    if key == 'cache_root':
        path.mkdir(parents=True, exist_ok=True)
    elif not path.exists():
        raise FileNotFoundError(path)
    print('PASS', key, path)
print('protocol=', cfg['protocol']['name'], 'version=', cfg['protocol']['version'])
print('severity=', float(cfg['experiment']['severity']))
print('allowed_stages=', cfg['protocol'].get('allowed_stages'))
from experiment_common.constants import (
    CACHE_SCHEMA_VERSION, DATA_CONTRACT_REVISION, TIME_FEATURE_REVISION,
    METRICS_REVISION, PREDICTION_CONTRACT_REVISION,
)
expected_contract = {
    'cache_schema_version': CACHE_SCHEMA_VERSION,
    'data_contract_revision': DATA_CONTRACT_REVISION,
    'time_feature_revision': TIME_FEATURE_REVISION,
    'metrics_revision': METRICS_REVISION,
    'prediction_contract_revision': PREDICTION_CONTRACT_REVISION,
}
for k,v in expected_contract.items():
    if str(cfg.get('contract', {}).get(k)) != str(v):
        raise RuntimeError(f'Contract config {k}={cfg.get("contract",{}).get(k)} != runtime {v}')
print('PASS contract schema/revisions', expected_contract)
stages = [a.stage] if a.stage else list(cfg['protocol'].get('allowed_stages', []))
for stage in stages:
    pm = load_protocol_stage_metadata(cfg, stage)
    print('PASS frozen Protocol', stage, pm['stage_hash'])
    print('CACHE_KEY', cache_dir(cfg, stage))
    if stage == 'evaluation':
        print('EVALUATION_ACCESS DIRECT_UNLOCKED')
print('PREFLIGHT PASS')
