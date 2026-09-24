#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, json, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from online_pstg_rag.config import load_config


def _stream(cmd, paths):
    for p in paths: p.parent.mkdir(parents=True,exist_ok=True)
    files=[p.open('a',encoding='utf-8') for p in paths]
    try:
        line='$ '+' '.join(map(str,cmd))+'\n'
        for f in files: f.write(line); f.flush()
        p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        assert p.stdout is not None
        for s in p.stdout:
            print(s,end='')
            for f in files: f.write(s); f.flush()
        rc=p.wait()
        if rc!=0: raise subprocess.CalledProcessError(rc,cmd)
    finally:
        for f in files: f.close()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development',choices=['development','formal']); ap.add_argument('--split',default=None)
    ap.add_argument('--project-root',default=None); ap.add_argument('--checkpoint',default=None); ap.add_argument('--max-queries',type=int,default=None)
    ap.add_argument('--memory-size',type=int,default=None); ap.add_argument('--min-memory-cases',type=int,default=None); ap.add_argument('--top-k',type=int,default=None)
    ap.add_argument('--alpha',type=float,default=None); ap.add_argument('--recency-tau',type=float,default=None); ap.add_argument('--no-gate',action='store_true')
    ap.add_argument('--scope',choices=['direct_mask','all'],default=None); ap.add_argument('--preflight-only',action='store_true')
    a=ap.parse_args(); cfg=load_config(ROOT,a.project_root)
    tag=f"online_pstg_rag_v010.{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir=Path(cfg['project']['log_root'])/cfg['experiment']['subgraph']/a.stage/tag; log_dir.mkdir(parents=True,exist_ok=True)
    (log_dir/'run_meta.json').write_text(json.dumps({'tag':tag,'stage':a.stage,'argv':sys.argv,'created':time.strftime('%Y-%m-%d %H:%M:%S')},indent=2),encoding='utf-8')
    common=[]
    if a.split: common += ['--split',a.split]
    if a.project_root: common += ['--project-root',a.project_root]
    if a.checkpoint: common += ['--checkpoint',a.checkpoint]
    pre=[sys.executable,str(ROOT/'scripts'/'preflight.py'),'--stage',a.stage]+common
    _stream(pre,[log_dir/'00_preflight.log',log_dir/'run.log'])
    if a.preflight_only:
        print(f'LOG_DIR={log_dir}'); return
    cmd=[sys.executable,str(ROOT/'scripts'/'run_rag.py'),'--stage',a.stage,'--run-tag',tag]+common
    for flag,val in [('--max-queries',a.max_queries),('--memory-size',a.memory_size),('--min-memory-cases',a.min_memory_cases),('--top-k',a.top_k),('--alpha',a.alpha),('--recency-tau',a.recency_tau),('--scope',a.scope)]:
        if val is not None: cmd += [flag,str(val)]
    if a.no_gate: cmd += ['--no-gate']
    _stream(cmd,[log_dir/'01_online_rag_stream.log',log_dir/'run.log'])
    print(f'LOG_DIR={log_dir}')
if __name__=='__main__': main()
