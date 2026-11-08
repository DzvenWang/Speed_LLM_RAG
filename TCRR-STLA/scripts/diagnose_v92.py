#!/usr/bin/env python3
"""v9.2 independent, post-training mechanism audit on fixed chronological validation CHECK."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tcrr_stla.v92.diagnostics import diagnose_run
p=argparse.ArgumentParser();p.add_argument('--run-identity',required=True)
p.add_argument('--max-queries',type=int,default=256);p.add_argument('--batch-size',type=int,default=8)
p.add_argument('--bootstrap',type=int,default=200)
p.add_argument('--wrong-mask-queries',type=int,default=32)
a=p.parse_args()
diagnose_run(a.run_identity,a.max_queries,a.batch_size,a.bootstrap,a.wrong_mask_queries)
