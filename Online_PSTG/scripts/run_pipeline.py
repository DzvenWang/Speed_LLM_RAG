#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, json, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from online_pstg.config import load_config


def tee_run(cmd, log_path):
    log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('a',encoding='utf-8') as f:
        f.write('\n$ '+' '.join(map(str,cmd))+'\n'); f.flush()
        p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            print(line,end=''); f.write(line); f.flush()
        rc=p.wait()
        if rc!=0: raise subprocess.CalledProcessError(rc,cmd)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',default='development',choices=['development','formal']); ap.add_argument('--split',default=None)
    ap.add_argument('--variant',default='head',choices=['frozen','head','last_block_head','full'])
    ap.add_argument('--project-root',default=None); ap.add_argument('--checkpoint',default=None); ap.add_argument('--max-queries',type=int,default=None)
    ap.add_argument('--replay-size',type=int,default=None); ap.add_argument('--min-mature-cases',type=int,default=None); ap.add_argument('--steps-per-update',type=int,default=None)
    ap.add_argument('--update-every',type=int,default=None); ap.add_argument('--batch-size',type=int,default=None); ap.add_argument('--learning-rate',type=float,default=None)
    ap.add_argument('--preflight-only',action='store_true')
    a=ap.parse_args(); cfg=load_config(ROOT,a.project_root)
    tag=f"online_pstg_v010.{a.variant}.{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir=Path(cfg['project']['log_root'])/cfg['experiment']['subgraph']/a.stage/tag
    log_dir.mkdir(parents=True,exist_ok=True)
    meta={'tag':tag,'stage':a.stage,'variant':a.variant,'project_root':cfg['project']['root'],'argv':sys.argv,'created':time.strftime('%Y-%m-%d %H:%M:%S')}
    (log_dir/'run_meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    common=[]
    if a.split: common += ['--split',a.split]
    if a.project_root: common += ['--project-root',a.project_root]
    if a.checkpoint: common += ['--checkpoint',a.checkpoint]
    pre=[sys.executable,str(ROOT/'scripts'/'preflight.py'),'--stage',a.stage]+common
    tee_run(pre,log_dir/'00_preflight.log'); tee_run(pre,log_dir/'run.log') if False else None
    # copy preflight into master log without rerunning
    (log_dir/'run.log').write_text((log_dir/'00_preflight.log').read_text(encoding='utf-8'),encoding='utf-8')
    if a.preflight_only:
        print(f'LOG_DIR={log_dir}'); return
    cmd=[sys.executable,str(ROOT/'scripts'/'run_online.py'),'--stage',a.stage,'--variant',a.variant,'--run-tag',tag]+common
    for flag,val in [('--max-queries',a.max_queries),('--replay-size',a.replay_size),('--min-mature-cases',a.min_mature_cases),('--steps-per-update',a.steps_per_update),('--update-every',a.update_every),('--batch-size',a.batch_size),('--learning-rate',a.learning_rate)]:
        if val is not None: cmd += [flag,str(val)]
    # stream output to both stage log and master log
    stage_log=log_dir/'01_online_stream.log'
    stage_log.parent.mkdir(parents=True,exist_ok=True)
    with stage_log.open('a',encoding='utf-8') as sf,(log_dir/'run.log').open('a',encoding='utf-8') as mf:
        sf.write('\n$ '+' '.join(cmd)+'\n'); mf.write('\n$ '+' '.join(cmd)+'\n'); sf.flush(); mf.flush()
        p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            print(line,end=''); sf.write(line); mf.write(line); sf.flush(); mf.flush()
        rc=p.wait()
        if rc!=0: raise subprocess.CalledProcessError(rc,cmd)
    print(f'LOG_DIR={log_dir}')
if __name__=='__main__': main()
