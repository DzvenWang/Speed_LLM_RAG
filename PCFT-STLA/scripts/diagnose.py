#!/usr/bin/env python3
"""Run same-checkpoint evidence and action ablations; no independent no-retrieval training."""
from pathlib import Path
import argparse,json,subprocess,sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_MODES=['full', 'legacy_only', 'propagation_only', 'no_lag_alignment', 'no_prop_retrieval', 'observed_support_only', 'no_action_utility']

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--modes',nargs='+',default=DEFAULT_MODES)
    p.add_argument('--severities',nargs='+',type=int,default=[20,40,60,90])
    p.add_argument('--batch-size',type=int,default=8)
    args=p.parse_args()
    checkpoint=Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    collected=[]
    for mode in args.modes:
        cmd=[sys.executable,'-u',str(ROOT/'scripts/evaluate.py'),
             '--checkpoint',str(checkpoint),'--evidence-mode',mode,
             '--batch-size',str(args.batch_size),'--severities',*[str(v) for v in args.severities]]
        subprocess.check_call(cmd,cwd=ROOT)
        path=checkpoint.parent/'evaluation'/f'severity_summary.{mode}.csv'
        frame=pd.read_csv(path)
        # Evaluation writers use the exact Base_SupportBench MAE / RMSE / MAPE definitions.
        for col in ('MAE','RMSE','MAPE'):
            if col not in frame.columns:raise RuntimeError(f'Missing {col} in {path}')
        collected.append({'mode':mode,'n_severities':len(frame),
                          **{f'Avg_{col}':float(frame[col].mean()) for col in ('MAE','RMSE','MAPE')}})
    output=checkpoint.parent/'evaluation'/'v2_diagnostic_summary.csv'
    pd.DataFrame(collected).to_csv(output,index=False)
    print('V2_DIAGNOSTICS_PASS',output,flush=True)

if __name__=='__main__':main()
