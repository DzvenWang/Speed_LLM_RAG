#!/usr/bin/env python3
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
from experiment_common.cache import validate_cache

ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['development','formal','evaluation'],required=True); a=ap.parse_args()
cfg=load_yaml(ROOT/'configs/baseline.yaml'); sc=shared_contract_config(cfg)
ident=validate_cache(sc,a.stage); shared=cache_path(cfg,a.stage)
print('PASS Shared Experiment Contract',ident['contract_version']); print('PASS stage',a.stage); print('PASS cache path',shared); print('PASS Protocol stage hash',ident['protocol_stage_hash']); print('PASS time feature fingerprint',ident['time_feature_fingerprint']); print('CACHE VALIDATION PASS')
