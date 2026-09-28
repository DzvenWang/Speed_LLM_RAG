#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline')
if not ROOT.exists(): ROOT=Path(__file__).resolve().parents[1]
COMMON=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/ExperimentCommon')
if not COMMON.exists(): COMMON=ROOT.parent/'ExperimentCommon'
for p in [ROOT,COMMON]:
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from baseline.utils import load_yaml
from baseline.shared_cache import cache_path
from experiment_common.cache import load_cache_identity
cfg=load_yaml(ROOT/'configs/baseline.yaml')
print(f"Baseline status v{cfg['version']} — Shared Experiment Contract {cfg['shared_contract']['required_version']}")
for stage in ['development','formal']:
    c=cache_path(cfg,stage)
    try:
        x=load_cache_identity(c); ok=str(x['contract_version'])==str(cfg['shared_contract']['required_version']); print(('PASS' if ok else 'FAIL'),stage,'shared_cache',x['protocol_name'],x['cache_fingerprint'][:12],x['protocol_stage_hash'][:12])
    except Exception as e: print('MISSING/INVALID',stage,repr(e))
for sev in cfg['experiment']['evaluation_severities']:
    try:
        c=cache_path(cfg,'evaluation',int(sev)); x=load_cache_identity(c); ok=str(x['contract_version'])==str(cfg['shared_contract']['required_version']); print(('PASS' if ok else 'FAIL'),f'evaluation-s{sev}','shared_cache',x['protocol_name'],x['cache_fingerprint'][:12],x['protocol_stage_hash'][:12])
    except Exception as e: print('MISSING/INVALID',f'evaluation-s{sev}',repr(e))
fb=Path(cfg['project']['frozen_backbone_json']); print(('READY' if fb.exists() else 'NOT READY'),'frozen PatchSTG',fb)
