#!/usr/bin/env python3
"""Mandatory post-training evaluation hook. Never freezes a checkpoint."""
from pathlib import Path
import argparse, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
MODELS=['STAEformer','HimNet','BigST','STDN','PatchSTG','STLAformer']
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True,choices=MODELS)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--seed',type=int,default=10)
    ap.add_argument('--run-stamp',default=None)
    ap.add_argument('--config',default=str(ROOT/'configs/backbone.yaml'))
    a=ap.parse_args()
    cmd=[sys.executable,str(ROOT/'scripts/run_checkpoint_evaluations.py'),'--model',a.model,'--checkpoint',a.checkpoint,'--expected-seed',str(a.seed),'--severities','20','40','60','90','--config',a.config]
    if a.run_stamp: cmd += ['--run-stamp',a.run_stamp]
    print('POST_TRAINING_EVAL RUN',' '.join(cmd),flush=True)
    subprocess.check_call(cmd,cwd=ROOT)
    print(f'POST_TRAINING_EVAL_PASS model={a.model} seed={a.seed} checkpoint={a.checkpoint}',flush=True)
if __name__=='__main__': main()
