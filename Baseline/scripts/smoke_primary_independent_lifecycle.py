#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path

ROOT=next((p for p in Path(__file__).resolve().parents if p.name=='Baseline'),Path(__file__).resolve().parents[1])
PROJECT=ROOT.parent
sys.path.insert(0,str(ROOT/'scripts'))
from smoke_primary_group import TimestampTee, run, timestamp

METHODS=('GinAR','RAFT','RAST','SARAF','ReMFT')
TARGETS=('speed','flow')
SEVERITIES=(20,40,60,90)
RUNNERS={'GinAR':'ginar_runner.py','RAFT':'raft_runner.py','RAST':'rast_runner.py','SARAF':'saraf_runner.py','ReMFT':'remft_runner.py'}

def main():
    ap=argparse.ArgumentParser(description='Independent Formal lifecycle smoke: limited real training -> disk checkpoint -> fresh-process reload -> limited s20/s40/s60/s90 real inference.')
    ap.add_argument('--seed',type=int,default=10)
    ap.add_argument('--continue-on-error',action='store_true')
    ap.add_argument('--keep-artifacts',action='store_true',help='Keep _lifecycle_smoke checkpoints/predictions after success.')
    ap.add_argument('--eval-batches',type=int,default=2,help='Real evaluation batches per severity (default: 2).')
    a=ap.parse_args()
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    log=PROJECT/'Baseline_logs'/'smoke_primary'/'independent_lifecycle'/f'BaselineSmokeIndependentLifecycle.{stamp}.log'
    tee=TimestampTee(log); old=sys.stdout; sys.stdout=tee
    root=PROJECT/'Baseline_exps'/'_lifecycle_smoke'/stamp
    passed=0; failed=[]; t0=time.time()
    try:
        print(f'INDEPENDENT_FORMAL_LIFECYCLE_SMOKE_START cases=10 seed={a.seed}')
        print(f'MODE=limited_formal_training+disk_checkpoint+fresh_process_reload+limited_four_severity_inference eval_batches={a.eval_batches}')
        print(f'ARTIFACT_ROOT={root}')
        for target in TARGETS:
            for method in METHODS:
                label=f'{method}/{target}'; c0=time.time(); case=root/target/method; cp=case/'best.pt'
                print('='*100); print('LIFECYCLE CASE START',label); print('='*100)
                env=os.environ.copy(); env.update(SPEED_LLM_RAG_ROOT=str(PROJECT),TRAFFIDENT_TARGET=target,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',PYTHONIOENCODING='utf-8',BASELINE_LIFECYCLE_CHECKPOINT=str(cp))
                try:
                    run([sys.executable,'scripts/preflight.py','--stage','formal','--method',method],env,tee)
                    run([sys.executable,'scripts/preflight_formal.py','--stage','formal','--method',method,'--seed',str(a.seed)],env,tee)
                    if method=='RAST': run([sys.executable,'scripts/check_rast_runtime.py'],env,tee)
                    if method=='ReMFT': run([sys.executable,'scripts/check_remft_runtime.py'],env,tee)
                    cmd=[sys.executable,'-u',str(ROOT/'baseline'/'runners'/RUNNERS[method]),'--stage','formal','--seed',str(a.seed),'--smoke']
                    if method=='RAST': cmd += ['--variant','native']
                    run(cmd,env,tee)
                    if not cp.exists() or cp.stat().st_size==0: raise RuntimeError(f'Lifecycle checkpoint missing/empty: {cp}')
                    print(f'CHECKPOINT_SAVE_VERIFIED method={method} target={target} bytes={cp.stat().st_size}')
                    # Each evaluation is a new Python process: model construction and torch.load cross a real serialization boundary.
                    for sev in SEVERITIES:
                        run([sys.executable,'scripts/preflight.py','--stage','evaluation','--method',method,'--severity',str(sev)],env,tee)
                        run([sys.executable,'scripts/preflight_formal.py','--stage','evaluation','--method',method,'--seed',str(a.seed),'--severity',str(sev)],env,tee)
                        run([sys.executable,'-u','scripts/run_evaluation.py','--method',method,'--seed',str(a.seed),'--severity',str(sev),'--lifecycle-checkpoint',str(cp),'--lifecycle-max-batches',str(a.eval_batches)],env,tee)
                    passed+=1; print(f'LIFECYCLE CASE PASS {label} seconds={time.time()-c0:.1f}')
                except Exception as exc:
                    failed.append((label,repr(exc))); print(f'LIFECYCLE CASE FAIL {label}: {exc!r}')
                    if not a.continue_on_error: raise
        print('='*100)
        print(f'INDEPENDENT_FORMAL_LIFECYCLE_SMOKE_{"PASS" if not failed else "PARTIAL"} {passed}/10 seconds={time.time()-t0:.1f}')
        if failed:
            for x,e in failed: print('FAILED',x,e)
            raise SystemExit(1)
        if not a.keep_artifacts:
            shutil.rmtree(root,ignore_errors=True); print(f'LIFECYCLE_ARTIFACT_CLEANUP_PASS root={root}')
    finally:
        sys.stdout=old; tee.close(); print(f'[{timestamp()}] Aggregate lifecycle smoke log: {log}',flush=True)
if __name__=='__main__': main()
