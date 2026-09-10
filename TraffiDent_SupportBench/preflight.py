#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, sys
from pathlib import Path
from supportbench.config import discover_source_root, load_config


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default=None); args=ap.parse_args(); cfg=load_config(args.config)
    print('TraffiDent-SupportBench preflight')
    print('config:',cfg['_config_path'])
    for k,v in cfg['project'].items(): print(f'{k}: {v}')
    for mod in ['numpy','pandas','scipy','yaml','sklearn']:
        ok=importlib.util.find_spec(mod) is not None; print(('PASS' if ok else 'FAIL'),mod)
        if not ok: sys.exit(2)
    try:
        source=discover_source_root(cfg); print('PASS source_root',source)
    except Exception as e:
        print('FAIL source_root',e); sys.exit(2)
    sf=cfg['source']; failed=False
    for sg in cfg['source'].get('expected_subgraphs',[]):
        p=source/sg
        req=[sf['traffic_file'],sf['nodes_file'],sf['adjacency_file']]
        missing=[x for x in req if not (p/x).exists()]
        if missing: print('FAIL',sg,'missing',missing); failed=True
        else: print('PASS',sg)
    if failed: sys.exit(2)
    print('PREFLIGHT PASS')
if __name__=='__main__': main()
