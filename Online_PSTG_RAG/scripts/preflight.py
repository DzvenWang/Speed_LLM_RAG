#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, sys, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from online_pstg_rag.config import load_config
from online_pstg_rag.contract import assert_cache_ready
from online_pstg_rag.backbone import resolve_checkpoint
from online_pstg_rag.data import OnlineCache
from online_pstg_rag.utils import log


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development',choices=['development','formal']); ap.add_argument('--split',default=None)
    ap.add_argument('--project-root',default=None); ap.add_argument('--checkpoint',default=None)
    a=ap.parse_args(); cfg=load_config(ROOT,a.project_root)
    if a.split: cfg['experiment']['stream_split']=a.split
    ident=assert_cache_ready(cfg,a.stage,a.split); ds=OnlineCache(cfg,a.stage,a.split)
    ck_path,ck,source=resolve_checkpoint(cfg,a.stage,a.checkpoint)
    log(f"PASS project_root={cfg['project']['root']}")
    log(f"PASS cache stage={a.stage} split={ds.split} queries={len(ds)} fingerprint={ident.get('cache_fingerprint')}")
    log(f"PASS PatchSTG checkpoint={ck_path} source={source} seed={ck.get('seed')} mean={ck.get('mean')} std={ck.get('std')}")
    log(f"PASS causal contract H={ds.history} P={ds.horizon}; a case matures only at start+H+P")
    log(f"PASS rag memory_size={cfg['rag']['memory_size']} top_k={cfg['rag']['top_k']} alpha={cfg['rag']['alpha']}")
    log(f"PASS torch={torch.__version__} cuda={torch.cuda.is_available()}")
    log('PREFLIGHT PASS')
if __name__=='__main__': main()
