#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, subprocess, sys, itertools
ROOT=Path(__file__).resolve().parents[1]

def _vals(s,cast): return [cast(x.strip()) for x in s.split(',') if x.strip()]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development',choices=['development','formal']); ap.add_argument('--project-root',default=None)
    ap.add_argument('--checkpoint',default=None); ap.add_argument('--max-queries',type=int,default=None)
    ap.add_argument('--top-k',default='3,5,8'); ap.add_argument('--alpha',default='0.15,0.25,0.35'); ap.add_argument('--memory-size',default='256,512')
    a=ap.parse_args()
    for k,alpha,mem in itertools.product(_vals(a.top_k,int),_vals(a.alpha,float),_vals(a.memory_size,int)):
        cmd=[sys.executable,str(ROOT/'scripts'/'run_pipeline.py'),'--stage',a.stage,'--top-k',str(k),'--alpha',str(alpha),'--memory-size',str(mem)]
        if a.project_root: cmd += ['--project-root',a.project_root]
        if a.checkpoint: cmd += ['--checkpoint',a.checkpoint]
        if a.max_queries is not None: cmd += ['--max-queries',str(a.max_queries)]
        print('RUN',' '.join(cmd),flush=True); subprocess.run(cmd,check=True)
if __name__=='__main__': main()
