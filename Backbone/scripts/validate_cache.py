#!/usr/bin/env python3
from pathlib import Path
import argparse, sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from backbone.utils import load_yaml
from backbone.shared_cache import cache_path, validate_shared_cache
from backbone.identity import load_cache_identity, require_cache_version_compatibility

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['development','formal','evaluation'],required=True); ap.add_argument('--severity',type=int,choices=[20,40,60,90]); ap.add_argument('--allow-locked-evaluation',action='store_true'); ap.add_argument('--config',default=str(ROOT/'configs/backbone.yaml')); a=ap.parse_args()
    if a.stage=='evaluation' and a.severity is None: raise SystemExit('--severity required')
    cfg=load_yaml(a.config); ident=validate_shared_cache(cfg,a.stage,a.severity,allow_locked_evaluation=a.allow_locked_evaluation); require_cache_version_compatibility(cfg,ident,a.stage); p=cache_path(cfg,a.stage,a.severity)
    print('PASS Shared Experiment Contract',ident['shared_contract_version']); print('PASS cache schema',ident['cache_schema_version']); print('PASS protocol',ident['protocol_name'],ident['protocol_version']); print('PASS severity',ident['severity']); print('PASS stage',a.stage); print('PASS cache path',p); print('PASS Protocol stage hash',ident['protocol_stage_hash']); print('PASS time feature fingerprint',ident['time_feature_fingerprint']); print('CACHE VALIDATION PASS')
if __name__=='__main__': main()
