#!/usr/bin/env python3
"""Prepare the single shared ExperimentCache for Backbone."""
from pathlib import Path
import argparse, os, sys

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(os.environ.get('SPEED_LLM_RAG_ROOT', ROOT.parent)).expanduser().resolve()
COMMON = REPO_ROOT / 'ExperimentCommon'
for q in [ROOT, COMMON, REPO_ROOT / 'Protocol', REPO_ROOT / 'TraffiDent_SupportBench']:
    if str(q) not in sys.path:
        sys.path.insert(0, str(q))
from backbone.utils import load_yaml
from backbone.shared_cache import shared_contract_config, shared_stage, cache_path
from experiment_common.cache import prepare_cache, validate_cache

ap = argparse.ArgumentParser()
ap.add_argument('--stage', choices=['screening', 'formal', 'evaluation'], required=True)
ap.add_argument('--force', action='store_true')
ap.add_argument('--config', default=str(ROOT / 'configs' / 'backbone_selection.yaml'))
a = ap.parse_args()
cfg = load_yaml(a.config)
scfg = shared_contract_config(cfg)
ss = shared_stage(cfg, a.stage)
shared = prepare_cache(scfg, ss, force=a.force)
validate_cache(scfg, ss)
expected = cache_path(cfg, a.stage)
if shared.resolve() != expected.resolve():
    raise RuntimeError(f'Shared cache path mismatch: prepared={shared} expected={expected}')
print('SHARED_CACHE=' + str(shared))
print('CACHE READY')
