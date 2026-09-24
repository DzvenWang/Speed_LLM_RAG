#!/usr/bin/env python3
from pathlib import Path
import argparse, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development',choices=['development','formal']); ap.add_argument('--split',default=None)
    ap.add_argument('--variants',default='frozen,head,last_block_head,full'); ap.add_argument('--project-root',default=None); ap.add_argument('--checkpoint',default=None); ap.add_argument('--max-queries',type=int,default=None)
    a=ap.parse_args(); batch=time.strftime('%Y%m%d_%H%M%S')
    for v in [x.strip() for x in a.variants.split(',') if x.strip()]:
        cmd=[sys.executable,str(ROOT/'scripts'/'run_online.py'),'--stage',a.stage,'--variant',v,'--run-tag',f'online_pstg_v010.{batch}.{v}']
        if a.split: cmd += ['--split',a.split]
        if a.project_root: cmd += ['--project-root',a.project_root]
        if a.checkpoint: cmd += ['--checkpoint',a.checkpoint]
        if a.max_queries is not None: cmd += ['--max-queries',str(a.max_queries)]
        print('RUN',' '.join(cmd),flush=True); subprocess.run(cmd,check=True)
if __name__=='__main__': main()
