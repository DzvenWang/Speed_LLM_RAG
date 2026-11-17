#!/usr/bin/env python3
"""Isolated CPU synthetic E2E for v0.2.3 (dose ladder + view consistency enabled).

Runs train -> lock -> TEST -> report through scripts/v0_2/run.py with
configs/v0_2/corsta_v023.yaml in a throw-away synthetic project, then checks that
the new diagnostics were written.  No forecasting claims.
"""
from __future__ import annotations
import json,os,sys,subprocess,argparse
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
    cmd=[sys.executable,'-u',str(SRC/'scripts/v0_2/run.py'),
         '--target',target,'--dataset','TraffiDent_Net_06','--seed','10',
         '--backbone-checkpoint',str(ck),'--mask-proj-checkpoint',str(s0),
         '--config',str(SRC/'configs/v0_2/corsta_v023.yaml'),
         '--cpu','--smoke','--limit-train','8','--limit-val','48','--limit-test','4',
         '--workers','0','--stage','all']
    result=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=300)
    print(f'V023_SYNTH target={target} status={result.returncode}',flush=True)
    if result.returncode:
        print(result.stdout[-8000:],result.stderr[-3000:],flush=True)
        raise RuntimeError(f'v0.2.3 synthetic run failed: {root}')
    assert 'CORSTA_V023_DOSE_LADDER' in result.stdout,'dose ladder teacher rungs were not activated'
    run=root/f'CoRSTA_v023_exps/{target}/incomplete/TraffiDent_Net_06/smoke/seed10'
    for st in ('normal','anchor_cf','anchor_control'):
        assert (run/st/'best.pt').exists() and (run/st/'COMPLETE.json').exists(),st
    assert (run/'model_lock.json').exists() and (run/'summary.json').exists()
    for sev in (20,40,60,90):
        diag=json.loads((run/'test'/f's{sev}.json').read_text())['metric']['per_severity'][str(sev)]['cf_diagnostics']
        assert 'dose' in diag and 'consistency_bins' in diag,f'missing new diagnostics at s{sev}'
        assert len(diag['consistency_bins'])==4
    summary=json.loads((run/'summary.json').read_text())['macro_severities']
    assert set(summary)=={'backbone','normal','normal_cf_candidate','final'}
    print('V023_SYNTHETIC_E2E_PASS',target,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--target',choices=('speed','flow','both'),default='both')
    a=parser.parse_args()
    if a.target=='both':
        for target in ('speed','flow'):
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--target',target],
                           check=True,timeout=400,env={**os.environ,'OMP_NUM_THREADS':'1',
                           'MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    else:
        per_target(a.target)


if __name__=='__main__':main()
