from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)) if str(ROOT) not in sys.path else None
from fare.context import FAREContext
from fare.evaluation import evaluate_experts
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--target',required=True,choices=['speed','flow']); ap.add_argument('--severity',type=int,required=True,choices=[20,40,60,90]); ap.add_argument('--max-rows',type=int,default=None); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args(); ctx=FAREContext(a.target,run_mode='smoke' if a.smoke else 'formal'); evaluate_experts(ctx,a.severity,a.max_rows,a.smoke)
