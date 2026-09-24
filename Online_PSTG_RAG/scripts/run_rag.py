#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, sys, time, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from online_pstg_rag.config import load_config
from online_pstg_rag.contract import assert_cache_ready
from online_pstg_rag.backbone import load_patchstg
from online_pstg_rag.data import OnlineCache
from online_pstg_rag.stream import run_stream
from online_pstg_rag.utils import seed_all, log, write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',default='development',choices=['development','formal']); ap.add_argument('--split',default=None)
    ap.add_argument('--run-tag',default=None); ap.add_argument('--project-root',default=None); ap.add_argument('--checkpoint',default=None)
    ap.add_argument('--max-queries',type=int,default=None); ap.add_argument('--memory-size',type=int,default=None); ap.add_argument('--min-memory-cases',type=int,default=None)
    ap.add_argument('--top-k',type=int,default=None); ap.add_argument('--alpha',type=float,default=None); ap.add_argument('--recency-tau',type=float,default=None)
    ap.add_argument('--no-gate',action='store_true'); ap.add_argument('--scope',choices=['direct_mask','all'],default=None)
    a=ap.parse_args(); cfg=load_config(ROOT,a.project_root); seed_all(cfg['experiment']['seed'])
    if a.split: cfg['experiment']['stream_split']=a.split
    for key,val in [('memory_size',a.memory_size),('min_memory_cases',a.min_memory_cases),('top_k',a.top_k),('alpha',a.alpha),('recency_tau_steps',a.recency_tau),('correction_scope',a.scope)]:
        if val is not None: cfg['rag'][key]=val
    if a.no_gate: cfg['rag']['use_confidence_gate']=False
    assert_cache_ready(cfg,a.stage,a.split); ds=OnlineCache(cfg,a.stage,a.split)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model,ck_path,ck,source=load_patchstg(cfg,a.stage,device,a.checkpoint)
    tag=a.run_tag or f"online_pstg_rag_v010.{time.strftime('%Y%m%d_%H%M%S')}"
    out=Path(cfg['project']['exp_root'])/cfg['experiment']['subgraph']/a.stage/tag; out.mkdir(parents=True,exist_ok=True)
    write_json(out/'resolved_config.json',cfg)
    log(f"RUN stage={a.stage} split={ds.split} out={out}")
    summary=run_stream(cfg,ds,model,ck_path,ck,source,out,device,a.max_queries)
    log('='*90); log(f"FINAL base masked_MAE={summary['base']['masked_mae']:.6f}")
    log(f"FINAL RAG  masked_MAE={summary['rag']['masked_mae']:.6f} gain={summary['gains_pct']['masked_mae']:+.3f}%")
    od=summary['oracle_topk_diagnostic']; log(f"ORACLE TopK query gain={od['gain_pct']:+.3f}% on {od['queries']} eligible queries")
    log(f"SUMMARY={out/'summary.json'}"); log('='*90)
if __name__=='__main__': main()
