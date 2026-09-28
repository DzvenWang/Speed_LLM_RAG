#!/usr/bin/env python3
"""Run all 6 Backbone x 2 targets real-Formal smoke tests sequentially."""
from pathlib import Path
import argparse, os, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[1]
GENERIC=('STAEformer','HimNet','BigST','STDN','STLAformer')
ORDER=('STAEformer','HimNet','BigST','STDN','PatchSTG','STLAformer')
TARGETS=('speed','flow')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,default=10)
    ap.add_argument('--train-batches',type=int,default=2)
    ap.add_argument('--val-batches',type=int,default=1)
    ap.add_argument('--continue-on-error',action='store_true')
    a=ap.parse_args()
    if a.seed != 10: raise SystemExit('Formal smoke seed is fixed to 10')
    jobs=[(m,t) for m in ORDER for t in TARGETS]; failures=[]; t0=time.time()
    print(f'ALL_FORMAL_SMOKE_START total={len(jobs)} seed={a.seed}',flush=True)
    for i,(model,target) in enumerate(jobs,1):
        print(f'[{i:02d}/{len(jobs)}] START model={model} target={target}',flush=True)
        env=os.environ.copy(); env['TRAFFIDENT_TARGET']=target; env['PYTHONUNBUFFERED']='1'
        if model=='PatchSTG':
            cmd=[sys.executable,'-u',str(ROOT/'scripts/run_experiment.py'),'--target',target,'--stage','formal','--seed',str(a.seed),'--smoke','--smoke-train-batches',str(a.train_batches),'--smoke-val-batches',str(a.val_batches)]
        else:
            cmd=[sys.executable,'-u',str(ROOT/'scripts/run_model_formal.py'),'--model',model,'--target',target,'--seed',str(a.seed),'--smoke','--smoke-train-batches',str(a.train_batches),'--smoke-val-batches',str(a.val_batches)]
        rc=subprocess.run(cmd,cwd=ROOT,env=env).returncode
        if rc==0: print(f'[{i:02d}/{len(jobs)}] PASS model={model} target={target}',flush=True)
        else:
            failures.append((model,target,rc)); print(f'[{i:02d}/{len(jobs)}] FAIL model={model} target={target} rc={rc}',flush=True)
            if not a.continue_on_error: break
    if failures:
        print('ALL_FORMAL_SMOKE_FAIL failures='+','.join(f'{m}:{t}:rc{r}' for m,t,r in failures),flush=True); raise SystemExit(1)
    print(f'ALL_FORMAL_SMOKE_PASS {len(jobs)}/{len(jobs)} seconds={time.time()-t0:.1f}',flush=True)
if __name__=='__main__': main()
