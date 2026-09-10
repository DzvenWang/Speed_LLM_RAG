#!/usr/bin/env python3
from pathlib import Path
import json, sys
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Protocol'); sys.path.insert(0,str(ROOT))
from shared_protocol.utils import load_yaml
cfg=load_yaml(ROOT/'configs/protocol.yaml')
base=Path(cfg['project']['artifact_root'])/cfg['experiment']['dataset']/cfg['protocol']['name']
print('Protocol:',base)
for stage in ['development','formal','evaluation']:
    sd=base/stage; frozen=sd/'FROZEN.json'; mf=sd/'stage_manifest.json'
    if frozen.exists() and mf.exists():
        j=json.loads(mf.read_text()); print(stage,'FROZEN',j['stage_hash'],j['splits'])
    else: print(stage,'NOT_BUILT')
print('evaluation_authorized=',(base/'EVALUATION_AUTHORIZED.json').exists())
