#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, time, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from online_pstg.config import load_config
from online_pstg.contract import assert_cache_ready
from online_pstg.backbone import load_patchstg
from online_pstg.data import OnlineCache
from online_pstg.stream import run_stream
from online_pstg.utils import seed_all, log, write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',default='development',choices=['development','formal']); ap.add_argument('--split',default=None)
    ap.add_argument('--variant',default='head',choices=['frozen','head','last_block_head','full'])
    ap.add_argument('--run-tag',default=None); ap.add_argument('--project-root',default=None); ap.add_argument('--checkpoint',default=None)
    ap.add_argument('--max-queries',type=int,default=None); ap.add_argument('--replay-size',type=int,default=None)
    ap.add_argument('--min-mature-cases',type=int,default=None); ap.add_argument('--steps-per-update',type=int,default=None)
    ap.add_argument('--update-every',type=int,default=None); ap.add_argument('--batch-size',type=int,default=None); ap.add_argument('--learning-rate',type=float,default=None)
    a=ap.parse_args(); cfg=load_config(ROOT,a.project_root); seed_all(cfg['experiment']['seed'])
    if a.split: cfg['experiment']['stream_split']=a.split
    for key,val in [('replay_size',a.replay_size),('min_mature_cases',a.min_mature_cases),('steps_per_update',a.steps_per_update),('update_every_queries',a.update_every),('batch_size',a.batch_size)]:
        if val is not None: cfg['online'][key]=val
    if a.learning_rate is not None: cfg['variants'][a.variant]['learning_rate']=a.learning_rate
    assert_cache_ready(cfg,a.stage,a.split); ds=OnlineCache(cfg,a.stage,a.split)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model,ck_path,ck,source=load_patchstg(cfg,a.stage,device,a.checkpoint)
    tag=a.run_tag or f"online_pstg_v010.{a.variant}.{time.strftime('%Y%m%d_%H%M%S')}"
    out=Path(cfg['project']['exp_root'])/cfg['experiment']['subgraph']/a.stage/tag; out.mkdir(parents=True,exist_ok=True)
    write_json(out/'resolved_config.json',cfg)
    log(f"RUN variant={a.variant} stage={a.stage} split={ds.split} out={out}")
    summary=run_stream(cfg,ds,model,ck_path,ck,source,a.variant,out,device,a.max_queries)
    log('='*80); log(f"FINAL static masked_MAE={summary['static']['masked_mae']:.6f}")
    log(f"FINAL online masked_MAE={summary['online']['masked_mae']:.6f} gain={summary['gains_pct']['masked_mae']:+.3f}%")
    log(f"SUMMARY={out/'summary.json'}"); log('='*80)
if __name__=='__main__': main()
