#!/usr/bin/env python3
"""Orchestrate full training -> independently trained OOF control -> calibrated best -> test."""
import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--dataset',default='TraffiDent_Net_06');p.add_argument('--target',default='speed');p.add_argument('--seed',type=int,default=10);p.add_argument('--backbone-checkpoint');p.add_argument('--skip-test',action='store_true')
a=p.parse_args()
me=Path(__file__).resolve().parent
from datetime import datetime
root=me.parents[1]
run=root/'FARE_exps'/a.target/a.dataset/'formal'/f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_seed{a.seed}'
cmd=[sys.executable,str(me/'train.py'),'--dataset',a.dataset,'--target',a.target,'--seed',str(a.seed),'--run-dir',str(run)]
if a.backbone_checkpoint:cmd+=['--backbone-checkpoint',a.backbone_checkpoint]
subprocess.check_call(cmd)
subprocess.check_call([sys.executable,str(me/'build_oof.py'),'--checkpoint',str(run/'candidate_best.pt')])
subprocess.check_call([sys.executable,str(me/'calibrate.py'),'--checkpoint',str(run/'controller_best.pt')])
if not a.skip_test:subprocess.check_call([sys.executable,str(me/'evaluate.py'),'--checkpoint',str(run/'best.pt')])
print(f'FARE_FULL_PIPELINE_PASS run_dir={run}',flush=True)
