#!/usr/bin/env python3
from pathlib import Path
import ast,yaml
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline') if Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline').exists() else Path(__file__).resolve().parents[1]
cfg=yaml.safe_load((ROOT/'configs/baseline.yaml').read_text())
for method,file in [('GinAR','ginar_runner.py'),('RAFT','raft_runner.py'),('SARAF','saraf_runner.py'),('ImputeFormer','imputeformer_runner.py'),('RAST','rast_runner.py')]:
 p=ROOT/'baseline'/'runners'/file;ast.parse(p.read_text());assert method in cfg['runners'];print('PASS runner static contract',method,p)
print('NEW MAIN RUNNER STATIC SMOKE PASS')
