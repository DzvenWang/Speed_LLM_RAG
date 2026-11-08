#!/usr/bin/env python3
"""Isolated CPU synthetic E2E for v0.2.2; no real forecasting claims."""
from __future__ import annotations
import os,sys,subprocess,argparse
from pathlib import Path
SRC=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SRC/'tests/v0_2'))
from integration_synthetic_v2 import synthetic_project


def per_target(target):
    root,ck,s0=synthetic_project(target)
    env=os.environ.copy()
    env.update(SPEED_LLM_RAG_ROOT=str(root),BACKBONE_MODE='incomplete',
               OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
               PYTHONPATH=str(SRC)+os.pathsep+str(root/'Backbone'))
    base=[sys.executable,'-u',str(SRC/'scripts/v0_2/run.py'),
          '--target',target,'--dataset','TraffiDent_Net_06','--seed','10',
          '--backbone-checkpoint',str(ck),'--mask-proj-checkpoint',str(s0),
          '--cpu','--smoke','--limit-train','8','--limit-val','48','--limit-test','4','--workers','0']
    # 'all' performs train -> lock -> TEST in the isolated synthetic sandbox.
    for stage in ('all',):
        result=subprocess.run(base+['--stage',stage],env=env,capture_output=True,text=True,timeout=120)
        print(f'V022_SYNTH target={target} stage={stage} status={result.returncode}',flush=True)
        if result.returncode:
            print(result.stdout[-6000:],result.stderr[-1500:],flush=True)
            raise RuntimeError(f'v0.2.2 synthetic stage failed: {root}')
        assert all(line.startswith('[') for line in result.stdout.splitlines())
        assert all(line.startswith('[') for line in result.stderr.splitlines())
    run=root/f'CoRSTA_v022_exps/{target}/incomplete/TraffiDent_Net_06/smoke/seed10'
    for st in ('normal','anchor_cf','anchor_control'):
        assert (run/st/'best.pt').exists()
        assert (run/st/'COMPLETE.json').exists()
    assert (run/'model_lock.json').exists()
    assert (run/'summary.json').exists()
    import json
    summary=json.loads((run/'summary.json').read_text())['macro_severities']
    assert set(summary)=={'backbone','normal','normal_cf_candidate','final'}
    print('V022_SYNTHETIC_E2E_PASS',target,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--target',choices=('speed','flow','both'),default='both')
    a=parser.parse_args()
    if a.target=='both':
        # Separate mock Backbone imports and torch runtime between targets.
        for target in ('speed','flow'):
            cmd=[sys.executable,str(Path(__file__).resolve()),'--target',target]
            subprocess.run(cmd,check=True,timeout=160,env={**os.environ,'OMP_NUM_THREADS':'1',
                'MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    else:
        per_target(a.target)

if __name__=='__main__':main()
