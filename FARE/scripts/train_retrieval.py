from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)) if str(ROOT) not in sys.path else None
from fare.context import FAREContext
from fare.training import train_retrieval
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--target',required=True,choices=['speed','flow']); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args(); train_retrieval(FAREContext(a.target,run_mode='smoke' if a.smoke else 'formal'),a.smoke)
