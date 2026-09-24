#!/usr/bin/env python3
from __future__ import annotations
import argparse,os,subprocess,sys
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from oracle.config import load_config
from oracle.utils import log


def run_stage(name,cmd,log_root,env):
    path=log_root/f'{name}.log'
    log('='*80); log(f'STAGE START {name}'); log(f'COMMAND {subprocess.list2cmdline(cmd)}'); log(f'LOG {path}')
    with path.open('w',encoding='utf-8') as fout:
        proc=subprocess.Popen(cmd,cwd=str(ROOT),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                              text=True,encoding='utf-8',errors='replace',bufsize=1)
        assert proc.stdout is not None
        for raw in proc.stdout:
            line=raw.rstrip('\r\n'); print(line,flush=True); fout.write(line+'\n'); fout.flush()
        code=proc.wait()
    if code: raise SystemExit(f'LOCAL PIPELINE FAILED stage={name} exit={code}; log={path}')
    log(f'STAGE PASS {name}')


def main():
    ap=argparse.ArgumentParser(description='Run all PatchSTG incomplete-sensing experiments on one local machine/GPU.')
    ap.add_argument('--stage',default='development',choices=['development','formal'])
    ap.add_argument('--run-tag',default=None)
    ap.add_argument('--checkpoint',default=None)
    ap.add_argument('--skip-preflight',action='store_true')
    ap.add_argument('--preflight-only',action='store_true')
    a=ap.parse_args()
    cfg=load_config(ROOT); seed=int(cfg['experiment']['seed'])
    tag=a.run_tag or datetime.now().strftime('%Y%m%d_%H%M%S')+f'_patchstg_observation_seed{seed}_local'
    log_root=Path(cfg['project']['log_root'])/'local'/tag; log_root.mkdir(parents=True,exist_ok=True)
    result_root=Path(cfg['project']['exp_root'])/cfg['experiment']['subgraph']/a.stage/tag
    env=os.environ.copy(); env.setdefault('PYTHONUNBUFFERED','1'); env.setdefault('PYTHONIOENCODING','utf-8')
    common=['--stage',a.stage,'--run-tag',tag]
    checkpoint=[] if not a.checkpoint else ['--checkpoint',str(Path(a.checkpoint).expanduser().resolve())]
    log(f'LOCAL PatchSTG observation experiments; python={sys.executable}')
    log(f'WORKSPACE_ROOT={cfg["project"]["root"]}'); log(f'RUN_TAG={tag}')
    stages=[('00_smoke',[sys.executable,str(ROOT/'tests'/'smoke_oracle.py')])]
    if not a.skip_preflight: stages.append(('01_preflight',[sys.executable,str(ROOT/'scripts'/'preflight.py'),'--stage',a.stage,*checkpoint]))
    if a.preflight_only:
        for name,cmd in stages: run_stage(name,cmd,log_root,env)
        log('LOCAL PREFLIGHT PASS'); return
    stages.extend([
        ('10_mask_aware',[sys.executable,str(ROOT/'scripts'/'run_observation_aware.py'),*common,'--variant','mask_aware',*checkpoint]),
        ('20_state_aux',[sys.executable,str(ROOT/'scripts'/'run_observation_aware.py'),*common,'--variant','state_aux',*checkpoint]),
    ])
    for name,cmd in stages: run_stage(name,cmd,log_root,env)
    mask_ckpt=result_root/'mask_aware'/'best.pt'; aux_ckpt=result_root/'state_aux'/'best.pt'
    coverage=[sys.executable,str(ROOT/'scripts'/'run_coverage_curve.py'),*common,*checkpoint,
              '--model',f'mask_aware={mask_ckpt}','--model',f'state_aux={aux_ckpt}']
    run_stage('30_coverage_curve',coverage,log_root,env)
    log('='*80); log('LOCAL PIPELINE PASS'); log(f'RESULT_ROOT={result_root}'); log(f'LOG_ROOT={log_root}')


if __name__=='__main__': main()
