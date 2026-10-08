#!/usr/bin/env python3
"""Run inference ablations on a fixed checkpoint; not a substitute for retrained ablations."""
import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--split',default='val',choices=['val','test']);p.add_argument('--severities',nargs='+',default=['20','40','60','90'])
a=p.parse_args()
for mode in ['full','retrieval_off','global_retrieval','no_mask_alignment','random_memory','state_only','innovation_only','no_risk']:
    cmd=[sys.executable,str(Path(__file__).with_name('evaluate.py')),'--checkpoint',a.checkpoint,'--split',a.split,'--mode',mode]
    if a.split=='test':cmd+=['--severities',*a.severities]
    print('FARE_ABLATION_START',mode,flush=True);subprocess.check_call(cmd)
print('FARE_ABLATION_PASS',flush=True)
