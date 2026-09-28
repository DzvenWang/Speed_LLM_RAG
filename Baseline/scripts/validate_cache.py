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
from baseline.shared_cache import shared_contract_config, cache_path, EVALUATION_SEVERITIES
from experiment_common.cache import validate_cache

ap=argparse.ArgumentParser()
ap.add_argument('--stage',choices=['development','formal','evaluation'],required=True)
ap.add_argument('--severity',type=int,choices=EVALUATION_SEVERITIES)
a=ap.parse_args()
if a.stage=='evaluation' and a.severity is None:
    raise SystemExit('--severity is required for evaluation')
cfg=load_yaml(ROOT/'configs/baseline.yaml')
sc=shared_contract_config(cfg,a.stage,a.severity)
ident=validate_cache(sc,a.stage)
shared=cache_path(cfg,a.stage,a.severity)
print('PASS Shared Experiment Contract',ident['contract_version'])
print('PASS stage',a.stage)
if a.stage=='evaluation': print('PASS severity',a.severity)
print('PASS cache path',shared)
print('PASS protocol',ident['protocol_name'],ident['protocol_version'])
print('PASS Protocol stage hash',ident['protocol_stage_hash'])
print('PASS time feature fingerprint',ident['time_feature_fingerprint'])
print('CACHE VALIDATION PASS')
