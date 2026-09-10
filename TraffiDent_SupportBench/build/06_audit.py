#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from supportbench.audit import audit_subgraph
from supportbench.config import load_config
from supportbench.utils import ensure_dir, write_json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default=None); ap.add_argument('--subgraph',action='append',default=[]); args=ap.parse_args(); cfg=load_config(args.config)
    root=Path(cfg['project']['benchmark_root']); diag=ensure_dir(cfg['project']['diagnostics_root']); items=sorted(root.glob('TraffiDent_Net_*'))
    if args.subgraph: items=[x for x in items if x.name in set(args.subgraph)]
    failed=False
    for x in items:
        r=audit_subgraph(x,args.config); write_json(diag/x.name/'audit.json',r); print(json.dumps(r,indent=2)); failed |= r['status']!='PASS'
    if failed: sys.exit(2)
if __name__=='__main__': main()
