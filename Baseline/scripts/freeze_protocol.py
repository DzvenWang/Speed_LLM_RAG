#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys, subprocess
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline'); sys.path.insert(0,str(ROOT))
from baseline.utils import load_yaml, write_json
from baseline.fingerprint import sha256_file, canonical_sha256
from baseline.registry import METHODS, ORDER
from baseline.run_identity import load_cache_identity
p=argparse.ArgumentParser(); p.add_argument('--note',default='Formal baseline protocol frozen after validation-only selection.'); a=p.parse_args()
cfg=load_yaml(ROOT/'configs/baseline.yaml'); exp=Path(cfg['project']['exp_root']); formal=exp/'_cache'/cfg['experiment']['subgraph']/'formal'/'cache_manifest.json'
subprocess.check_call([cfg['project']['python'],str(ROOT/'scripts/validate_cache.py'),'--stage','formal'])
if not formal.exists(): raise SystemExit('Formal cache must exist before protocol freeze.')
fb=Path(cfg['project']['frozen_backbone_json'])
identity=load_cache_identity(formal.parent,cfg['version'])
obj={'baseline_version':'0.3.1','formal_run_identity':identity,'config_sha256':sha256_file(ROOT/'configs/baseline.yaml'),'formal_cache_manifest':json.loads(formal.read_text()),'official_sources':{m:{'commit':METHODS[m]['commit'],'remote':METHODS[m]['repo']} for m in ORDER},'frozen_backbone_json':json.loads(fb.read_text()) if fb.exists() else None,'note':a.note}
obj['freeze_fingerprint']=canonical_sha256(obj)
out=exp/'FROZEN_BASELINE_PROTOCOL.json'; write_json(out,obj); print('wrote',out); print('FREEZE_FINGERPRINT='+obj['freeze_fingerprint'])
