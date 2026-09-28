#!/usr/bin/env python3
from pathlib import Path
import ast,yaml
ROOT=next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd()) if next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd()).exists() else Path(__file__).resolve().parents[1]
cfg=yaml.safe_load((ROOT/'configs/baseline.yaml').read_text())
for method,file in [('GinAR','ginar_runner.py'),('RAFT','raft_runner.py'),('SARAF','saraf_runner.py'),('ImputeFormer','imputeformer_runner.py'),('LRTC-TMCP','lrtc_tmcp_runner.py'),('RAST','rast_runner.py'),('ReMFT','remft_runner.py')]:
    p=ROOT/'baseline'/'runners'/file;ast.parse(p.read_text());assert method in cfg['runners'];print('PASS runner static contract',method,p)
print('PRIMARY RUNNER STATIC SMOKE PASS')
