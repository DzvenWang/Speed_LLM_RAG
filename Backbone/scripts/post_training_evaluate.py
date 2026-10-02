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
    # Resolve paths before changing cwd for the child process. This makes both
    # repository-relative inputs (Backbone/configs/...) and Backbone-relative
    # inputs (configs/...) safe for direct/local and PBS invocations.
    config_path=Path(a.config).expanduser()
    if not config_path.is_absolute():
        cwd_candidate=(Path.cwd()/config_path).resolve()
        root_candidate=(ROOT/config_path).resolve()
        config_path=cwd_candidate if cwd_candidate.exists() else root_candidate
    else:
        config_path=config_path.resolve()
    if not config_path.is_file():
        raise SystemExit(f'Config not found: {config_path}')
    checkpoint_path=Path(a.checkpoint).expanduser()
    if not checkpoint_path.is_absolute(): checkpoint_path=(Path.cwd()/checkpoint_path).resolve()
    else: checkpoint_path=checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise SystemExit(f'Checkpoint not found: {checkpoint_path}')
    cmd=[sys.executable,str(ROOT/'scripts/run_checkpoint_evaluations.py'),'--model',a.model,'--checkpoint',str(checkpoint_path),'--expected-seed',str(a.seed),'--severities','20','40','60','90','--config',str(config_path)]
    if a.run_stamp: cmd += ['--run-stamp',a.run_stamp]
    print('POST_TRAINING_EVAL RUN',' '.join(cmd),flush=True)
    subprocess.check_call(cmd,cwd=ROOT)
    print(f'POST_TRAINING_EVAL_PASS model={a.model} seed={a.seed} checkpoint={checkpoint_path}',flush=True)
if __name__=='__main__': main()
