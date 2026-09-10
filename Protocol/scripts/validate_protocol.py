#!/usr/bin/env python3
from pathlib import Path
import argparse, sys
ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Protocol')
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from shared_protocol.builder import validate_stage
from shared_protocol.utils import load_yaml
p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['development','formal','evaluation'],required=True); a=p.parse_args()
cfg=load_yaml(ROOT/'configs/protocol.yaml')
m=validate_stage(cfg,a.stage)
print('PASS',a.stage,m['stage_hash'])
print('PROTOCOL VALIDATION PASS')
