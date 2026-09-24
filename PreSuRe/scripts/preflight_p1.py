#!/usr/bin/env python3
from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from presure.p1_config import load_config
from presure.contract import assert_cache_ready
from presure.utils import log

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',default='development',choices=['development','formal']);a=ap.parse_args();cfg=load_config(ROOT)
    log(f'P1 PREFLIGHT version={cfg["version"]} workspace={cfg["project"]["root"]}')
    for key in ('backbone_root','backbone_exp_root','common_root','supportbench_root','benchmark_root'):
        p=Path(cfg['project'][key]);
        if not p.exists(): raise FileNotFoundError(f'{key}: {p}')
        log(f'PASS {key}={p}')
    ident=assert_cache_ready(cfg,a.stage);log(f'PASS ExperimentCache stage={a.stage} fingerprint={ident.get("cache_fingerprint")}')
    log('PASS P1 scientific contract: Frozen PatchSTG; Train-only memory; strict episode-end-before-query; no query hidden history; no query future in retrieval; historical future used only when already occurred and future_valid.')
if __name__=='__main__':main()
