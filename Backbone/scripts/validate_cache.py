#!/usr/bin/env python3
from pathlib import Path
import argparse, sys
GADI_ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone')
ROOT=GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[1]
COMMON=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/ExperimentCommon')
if not COMMON.exists(): COMMON=ROOT.parent/'ExperimentCommon'
for q in [ROOT,COMMON,ROOT.parent/'Protocol',ROOT.parent/'TraffiDent_SupportBench']:
    if str(q) not in sys.path: sys.path.insert(0,str(q))
from backbone.utils import load_yaml
from backbone.shared_cache import shared_contract_config, shared_stage, cache_path
from experiment_common.cache import validate_cache

ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['screening','formal','evaluation'],required=True); ap.add_argument('--config',default=str(ROOT/'configs/backbone_selection.yaml')); a=ap.parse_args()
cfg=load_yaml(a.config); scfg=shared_contract_config(cfg); ss=shared_stage(cfg,a.stage)
ident=validate_cache(scfg,ss); shared=cache_path(cfg,a.stage)
print('PASS Shared Experiment Contract',ident['contract_version']); print('PASS stage',ss); print('PASS cache path',shared); print('PASS Protocol stage hash',ident['protocol_stage_hash']); print('PASS time feature fingerprint',ident['time_feature_fingerprint']); print('CACHE VALIDATION PASS')
