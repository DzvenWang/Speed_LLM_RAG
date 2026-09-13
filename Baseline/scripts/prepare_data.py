#!/usr/bin/env python3
"""Prepare the single shared ExperimentCache for Baseline.

Baseline_exps contains method/run artifacts only. All benchmark arrays are read
directly from ExperimentCache; no consumer-local cache alias is created.
"""
from pathlib import Path
import argparse,sys
GADI_ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline')
ROOT=GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[1]
COMMON=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/ExperimentCommon')
if not COMMON.exists(): COMMON=ROOT.parent/'ExperimentCommon'
for q in [ROOT,COMMON,ROOT.parent/'Protocol',ROOT.parent/'TraffiDent_SupportBench']:
    if str(q) not in sys.path: sys.path.insert(0,str(q))
from baseline.utils import load_yaml
from baseline.shared_cache import shared_contract_config, cache_path
from experiment_common.cache import prepare_cache,validate_cache

ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['development','formal','evaluation'],default='development'); ap.add_argument('--force',action='store_true'); a=ap.parse_args()
cfg=load_yaml(ROOT/'configs/baseline.yaml'); sc=shared_contract_config(cfg)
shared=prepare_cache(sc,a.stage,force=a.force); validate_cache(sc,a.stage)
expected=cache_path(cfg,a.stage)
if shared.resolve()!=expected.resolve():
    raise RuntimeError(f'Shared cache path mismatch: prepared={shared} expected={expected}')
print('SHARED_CACHE='+str(shared)); print('CACHE READY')
