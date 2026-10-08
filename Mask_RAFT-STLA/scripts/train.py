#!/usr/bin/env python3
from pathlib import Path
import argparse, datetime, os, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from mask_raft_stla.config import load_config, cache_path, resolve_stla_checkpoint
from mask_raft_stla.trainer import train

def ensure_retrieval(cfg,split,smoke=False):
    tag='formal_smoke' if smoke else 'formal'
    out=Path(cfg['project']['retrieval_root'])/cfg['experiment']['dataset']/tag/f'{split}.mask_raft_topk.npz'
    if not out.exists():
        cmd=[sys.executable,'-u',str(ROOT/'scripts/build_retrieval_cache.py'),'--split',split,'--target',cfg['experiment']['target']]
        if smoke: cmd.append('--smoke')
        subprocess.check_call(cmd,cwd=ROOT)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--target',choices=['speed','flow'])
    ap.add_argument('--seed',type=int,default=10)
    ap.add_argument('--backbone-checkpoint')
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--run-stamp')
    ap.add_argument('--config',default=None)
    ap.add_argument('--batch-size',type=int)
    ap.add_argument('--num-workers',type=int)
    ap.add_argument('--skip-evaluate',action='store_true',help='Skip automatic formal s20/s40/s60/s90 evaluation after training')
    a=ap.parse_args()
    if a.target: os.environ['TRAFFIDENT_TARGET']=a.target
    cfg=load_config(a.config); cfg['experiment']['seed']=int(a.seed)
    if a.batch_size: cfg['training']['batch_size']=int(a.batch_size)
    if a.num_workers is not None: cfg['runtime']['num_workers']=int(a.num_workers)
    cache=cache_path(cfg,'formal'); ck=resolve_stla_checkpoint(cfg,a.backbone_checkpoint)
    rt=ensure_retrieval(cfg,'train',a.smoke); rv=ensure_retrieval(cfg,'val',a.smoke)
    stamp=a.run_stamp or datetime.datetime.now().strftime('%Y%m%d_%H%M%S'); stage='formal_smoke' if a.smoke else 'formal'
    run=Path(cfg['project']['exp_root'])/cfg['experiment']['dataset']/stage/f'{stamp}_seed{a.seed}'
    best=train(cfg,cache,ck,rt,rv,run,a.smoke); print(f'BEST_CHECKPOINT={best}',flush=True)
    if a.smoke:
        print('AUTO_EVALUATE_SKIP reason=smoke',flush=True)
    elif a.skip_evaluate:
        print('AUTO_EVALUATE_SKIP reason=cli',flush=True)
    else:
        severities=[str(int(x)) for x in cfg['experiment'].get('severities',[20,40,60,90])]
        cmd=[sys.executable,'-u',str(ROOT/'scripts/evaluate.py'),'--checkpoint',str(best),'--target',cfg['experiment']['target'],'--severities',*severities]
        if a.config: cmd.extend(['--config',a.config])
        print(f'AUTO_EVALUATE_START checkpoint={best} severities={",".join(severities)}',flush=True)
        subprocess.check_call(cmd,cwd=ROOT)
        print(f'AUTO_EVALUATE_PASS checkpoint={best}',flush=True)
if __name__=='__main__': main()
