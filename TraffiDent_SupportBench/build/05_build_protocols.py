#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from supportbench.config import load_config
from supportbench.protocols import write_default_protocols
from supportbench.utils import log

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default=None); ap.add_argument('--subgraph',action='append',default=[]); args=ap.parse_args(); cfg=load_config(args.config)
    root=Path(cfg['project']['benchmark_root']); items=sorted(root.glob('TraffiDent_Net_*'))
    if args.subgraph: items=[x for x in items if x.name in set(args.subgraph)]
    for x in items: write_default_protocols(x,cfg); log(f'{x.name}: protocols written')
if __name__=='__main__': main()
