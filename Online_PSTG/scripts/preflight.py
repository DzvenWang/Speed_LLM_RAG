#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from online_pstg.config import load_config
from online_pstg.contract import assert_cache_ready, cache_dir
from online_pstg.backbone import load_patchstg
from online_pstg.data import OnlineCache
from online_pstg.updater import configure_trainable
from online_pstg.utils import log, seed_all


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development',choices=['development','formal']); ap.add_argument('--split',default=None)
    ap.add_argument('--project-root',default=None); ap.add_argument('--checkpoint',default=None)
    a=ap.parse_args(); cfg=load_config(ROOT,a.project_root); seed_all(cfg['experiment']['seed'])
    if a.split: cfg['experiment']['stream_split']=a.split
    log(f"PROJECT_ROOT={cfg['project']['root']}")
    ident=assert_cache_ready(cfg,a.stage,a.split); log(f"CACHE PASS dir={cache_dir(cfg,a.stage)} fingerprint={ident.get('cache_fingerprint')}")
    ds=OnlineCache(cfg,a.stage,a.split); starts=ds.starts[ds.order]
    dif=np.diff(starts) if len(starts)>1 else np.asarray([])
    log(f"STREAM PASS split={ds.split} queries={len(ds)} start=[{starts[0]},{starts[-1]}] median_stride={float(np.median(dif)) if len(dif) else 'n/a'}")
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model,path,ck,source=load_patchstg(cfg,a.stage,device,a.checkpoint)
    log(f"BACKBONE PASS checkpoint={path} source={source} parameters={sum(p.numel() for p in model.parameters())}")
    import copy
    for v in ['frozen','head','last_block_head','full']:
        m=copy.deepcopy(model); ps=configure_trainable(m,cfg['variants'][v]['trainable'])
        log(f"VARIANT {v} trainable_parameters={sum(p.numel() for p in ps)}")
    # Causal sanity: first later query must not mature a future whose end is after its origin.
    if len(ds)>=2:
        i0=ds.cache_index(0); i1=ds.cache_index(1)
        log(f"CAUSAL_SANITY first_future_end={ds.future_end(i0)} second_origin={ds.origin(i1)} mature_at_second={ds.future_end(i0)<=ds.origin(i1)}")
    log('PREFLIGHT PASS')
if __name__=='__main__': main()
