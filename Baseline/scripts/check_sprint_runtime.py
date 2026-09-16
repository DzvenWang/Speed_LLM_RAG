#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import yaml

GADI_ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline')
ROOT = GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[1]
COMMON = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/ExperimentCommon')
if not COMMON.exists():
    COMMON = ROOT.parent / 'ExperimentCommon'
SUPPORTBENCH = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/TraffiDent_SupportBench')
if not SUPPORTBENCH.exists():
    SUPPORTBENCH = ROOT.parent / 'TraffiDent_SupportBench'
for p in (ROOT, COMMON, SUPPORTBENCH):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

repo = ROOT / 'vendor' / 'SPRINT'
for candidate in (repo / 'src', repo):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from SPRINT_imputations import impute
from baseline.runners.sprint_runner import sprint_vendor_defaults
from baseline.sprint_causal_history import resolve_canonical_history_source
from baseline.runners.common import scale_stats
from baseline.shared_cache import cache_path

ap = argparse.ArgumentParser()
ap.add_argument('--stage', choices=['development', 'formal', 'evaluation'], default='development')
a = ap.parse_args()

sig = inspect.signature(impute)
for name in (
    'ranks', 'max_iter', 'hyp_prior_list', 'hy1_list', 'Batch_per', 'Batch_iter',
    'miss_type', 'missing_rate', 'missing_rate_val', 'zero_as_missing',
):
    if name not in sig.parameters:
        raise RuntimeError(f'Official SPRINT impute() missing required public parameter {name}: {sig}')

native = sprint_vendor_defaults(impute)
print('PASS SPRINT pinned vendor-native algorithm defaults', native)

cfg = yaml.safe_load((ROOT / 'configs/baseline.yaml').read_text())
c = cfg['runners']['SPRINT']
if not bool(c.get('use_vendor_native_defaults', False)):
    raise RuntimeError('SPRINT benchmark must keep pinned vendor-native algorithm defaults')
if int(c['historical_days']) < 2:
    raise RuntimeError('SPRINT historical_days must be >=2')
if int(c['day_steps']) != 288:
    raise RuntimeError('TraffiDent 5-minute SPRINT adapter requires day_steps=288')
if cfg['methods']['SPRINT']['table_group'] != 'specialized_reconstruction_subtrack':
    raise RuntimeError('SPRINT causal historical-day adaptation must not be reported in the strict H12 main table')

native_ranks = [int(v) for v in native['ranks']]
context_days = int(c['historical_days']) + 1
if native_ranks and context_days <= max(native_ranks):
    raise RuntimeError(
        'SPRINT causal context is too short for pinned vendor ranks: '
        f'context_days={context_days}, native_ranks={native_ranks}. '
        'Increase historical_days rather than shrinking the baseline model rank.'
    )

cache = cache_path(cfg, a.stage)
split = 'test' if a.stage == 'evaluation' else 'val'
mean, std = scale_stats(cache)
source = resolve_canonical_history_source(cfg, cache, split, mean=mean, std=std)
print('PASS SPRINT causal history source', {
    'stage': a.stage,
    'split': split,
    'speed_path': str(source.speed_path),
    'speed_channel': source.speed_channel,
    'valid_path': str(source.valid_path),
    'valid_channel': source.valid_channel,
    'historical_days': source.historical_days,
    'context_days': source.context_days,
    'day_steps': source.day_steps,
    'native_ranks': native_ranks,
})
print('SPRINT RUNTIME CONTRACT PASS', sig)
