#!/usr/bin/env python3
"""Prepare/validate shared ExperimentCache contracts consumed by Baseline Core-5."""
from pathlib import Path
import argparse,sys
GADI_ROOT=next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd())
ROOT=GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[1]
COMMON=(next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd())).parent / 'Base_SupportBench'
if not COMMON.exists(): COMMON=ROOT.parent/'Base_SupportBench'
for q in [ROOT,COMMON,ROOT.parent/'Protocol',ROOT.parent/'Base_SupportBench']:
    if str(q) not in sys.path: sys.path.insert(0,str(q))
from baseline.utils import load_yaml
from baseline.shared_cache import shared_contract_config, cache_path, EVALUATION_SEVERITIES
from experiment_common.cache import prepare_cache,validate_cache

ap=argparse.ArgumentParser()
ap.add_argument('--stage',choices=['development','formal','evaluation'],default='development')
ap.add_argument('--severity',type=int,choices=EVALUATION_SEVERITIES)
ap.add_argument('--all-severities',action='store_true')
ap.add_argument('--force',action='store_true')
a=ap.parse_args()
cfg=load_yaml(ROOT/'configs/baseline.yaml')

if a.stage!='evaluation' and (a.severity is not None or a.all_severities):
    raise SystemExit('--severity/--all-severities are evaluation-only')
severities=list(EVALUATION_SEVERITIES) if (a.stage=='evaluation' and a.all_severities) else ([a.severity] if a.stage=='evaluation' else [None])
if a.stage=='evaluation' and severities==[None]:
    raise SystemExit('evaluation requires --severity or --all-severities')
for sev in severities:
    sc=shared_contract_config(cfg,a.stage,sev)
    shared=prepare_cache(sc,a.stage,force=a.force)
    validate_cache(sc,a.stage)
    expected=cache_path(cfg,a.stage,sev)
    if shared.resolve()!=expected.resolve():
        raise RuntimeError(f'Shared cache path mismatch: prepared={shared} expected={expected}')
    print(('EVALUATION_SEVERITY=s'+str(sev) if sev is not None else 'TRAINING_CONTRACT=s40'))
    print('SHARED_CACHE='+str(shared))
print('CACHE READY')
