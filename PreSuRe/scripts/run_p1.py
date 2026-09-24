#!/usr/bin/env python3
from __future__ import annotations
import argparse,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def run(label,args,logf):
    print(f'=== {label} ===',flush=True); p=subprocess.Popen(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    with open(logf,'w',encoding='utf-8') as f:
        for line in p.stdout:
            from datetime import datetime
            s=f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {line.rstrip()}'
            print(s,flush=True); f.write(s+'\n');f.flush()
    rc=p.wait()
    if rc: raise SystemExit(f'{label} failed rc={rc}; log={logf}')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',default='development',choices=['development','formal']);ap.add_argument('--run-tag',default=None);ap.add_argument('--force-prepare',action='store_true');a=ap.parse_args()
    from datetime import datetime
    tag=a.run_tag or datetime.now().strftime('%Y%m%d_%H%M%S_p1_seed10')
    ws=Path(os.environ.get('PRESURE_WORKSPACE_ROOT',ROOT.parent));logs=ws/'PreSuRe_logs'/'p1'/tag;logs.mkdir(parents=True,exist_ok=True);py=sys.executable
    run('00 smoke',[py,'tests/smoke_p1.py'],logs/'00_smoke.log')
    run('02 preflight',[py,'scripts/preflight_p1.py','--stage',a.stage],logs/'02_preflight.log')
    prep=[py,'scripts/prepare_p1.py','--stage',a.stage]+(['--force'] if a.force_prepare else [])
    run('10 prepare',prep,logs/'10_prepare.log')
    for i,v in enumerate(('adapter_only','shuffled_support','history_only','history_future'),start=30):
        run(f'{i} train {v}',[py,'scripts/train_p1.py','--stage',a.stage,'--variant',v,'--run-tag',tag],logs/f'{i}_{v}.log')
    run('70 summary',[py,'scripts/summarize_p1.py','--stage',a.stage,'--run-tag',tag],logs/'70_summary.log')
    print(f'P1 COMPLETE run_tag={tag} logs={logs}',flush=True)
if __name__=='__main__':main()
