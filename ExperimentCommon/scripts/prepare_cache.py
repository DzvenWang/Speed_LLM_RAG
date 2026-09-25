#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
for p in [ROOT, REPO_ROOT / 'Protocol', REPO_ROOT / 'TraffiDent_SupportBench']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiment_common.cache import prepare_cache, cache_dir
from experiment_common.utils import load_yaml

p = argparse.ArgumentParser()
p.add_argument('--stage', choices=['development', 'formal', 'evaluation'], required=True)
p.add_argument('--force', action='store_true')
p.add_argument('--config', default=str(ROOT / 'configs' / 'experiment_contract.yaml'))
p.add_argument('--materialize-locked-evaluation', action='store_true', help='Allow pre-materializing frozen evaluation tensors while the Protocol consumer gate remains locked.')
a = p.parse_args()
cfg = load_yaml(a.config)
print(f'CONTRACT={a.config}', flush=True)
print(f'PROTOCOL={cfg["protocol"]["name"]}', flush=True)
print(f'SEVERITY={float(cfg["experiment"]["severity"]):.2f}', flush=True)
print(f'STAGE={a.stage}', flush=True)
prepare_cache(cfg, a.stage, force=a.force, allow_locked_evaluation_materialization=a.materialize_locked_evaluation)
print('CACHE_DIR=' + str(cache_dir(cfg, a.stage)), flush=True)
print('CACHE PREPARE PASS', flush=True)
