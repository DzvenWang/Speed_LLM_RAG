from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from fare.context import FAREContext
from fare.training import calibrate_structural
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--target',required=True,choices=['speed','flow']); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args(); ctx=FAREContext(a.target,run_mode='smoke' if a.smoke else 'formal'); calibrate_structural(ctx,a.smoke)
