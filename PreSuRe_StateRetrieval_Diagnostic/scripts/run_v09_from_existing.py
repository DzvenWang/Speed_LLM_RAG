#!/usr/bin/env python3
from pathlib import Path
import argparse, subprocess, sys, time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts._common import setup, run_dir


def run(cmd, logfile):
    logfile=Path(logfile); logfile.parent.mkdir(parents=True,exist_ok=True)
    print('>>',' '.join(map(str,cmd)),flush=True)
    with logfile.open('w',encoding='utf-8') as f:
        p=subprocess.Popen([str(x) for x in cmd],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in p.stdout:
            print(line,end='',flush=True); f.write(line); f.flush()
        rc=p.wait()
    if rc: raise subprocess.CalledProcessError(rc,cmd)


def main():
    ap=argparse.ArgumentParser(description='Run v0.9 fragment-completion diagnostic from an existing patch retrieval run')
    ap.add_argument('--stage',default='development',choices=['development','formal'])
    ap.add_argument('--run-tag',required=True)
    ap.add_argument('--checkpoint',default=None)
    ap.add_argument('--scopes',default='12')
    ap.add_argument('--top-k',type=int,default=50)
    ap.add_argument('--transport',default='affine_alignment')
    ap.add_argument('--eval-batch-size',type=int,default=64)
    a=ap.parse_args()
    cfg=setup(a.stage); rd=run_dir(cfg,a.stage,a.run_tag)
    retrieval=rd/'patch_retrieval'/'patch_retrieval_topk.npz'
    if not retrieval.exists(): raise FileNotFoundError(f'Missing existing retrieval: {retrieval}')
    logroot=Path(cfg['project']['log_root'])/cfg['experiment']['subgraph']/a.stage/a.run_tag
    common=['--stage',a.stage,'--run-tag',a.run_tag]
    cmd=[sys.executable,ROOT/'scripts'/'15_run_fragment_completion_diagnostic.py','--retrieval',retrieval,*common,
         '--scopes',a.scopes,'--top-k',str(a.top_k),'--transport',a.transport,'--eval-batch-size',str(a.eval_batch_size)]
    if a.checkpoint: cmd += ['--checkpoint',a.checkpoint]
    t=time.time(); run(cmd,logroot/'15_fragment_completion_diagnostic.log')
    run([sys.executable,ROOT/'scripts'/'16_summarize_fragment_diagnostic.py',*common],logroot/'16_fragment_summary.log')
    print(f'V09_FRAGMENT_DIAGNOSTIC_COMPLETE elapsed={time.time()-t:.1f}s',flush=True)

if __name__=='__main__': main()
