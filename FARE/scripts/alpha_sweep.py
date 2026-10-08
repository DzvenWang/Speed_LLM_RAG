#!/usr/bin/env python3
"""Validation alpha sweep from previously exported FARE val predictions.

Examples:
  python -u FARE/scripts/evaluate.py --checkpoint <run>/candidate_best.pt --split val --output <run>/val_diagnostics
  python -u FARE/scripts/alpha_sweep.py --predictions <run>/val_diagnostics/val_full_predictions.npz

Never select alpha from test. This utility rejects non-validation files by default.
"""
import argparse, json
from pathlib import Path
import numpy as np
import _bootstrap
from fare.alpha import DEFAULT_ALPHA,scalar_sweep,save_sweep


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',required=True)
    p.add_argument('--output')
    p.add_argument('--max-alpha',type=float,default=.6)
    p.add_argument('--alphas',type=float,nargs='+')
    p.add_argument('--rmse-weight',type=float,default=.05)
    a=p.parse_args();src=Path(a.predictions)
    if not src.name.startswith('val_'):
        raise ValueError('Only validation prediction files can be used to select alpha (expected val_*.npz).')
    z=np.load(src,allow_pickle=False)
    for key in ('base','candidate','target','valid'):
        if key not in z:raise ValueError(f'Missing field {key} in {src}')
    alphas=a.alphas if a.alphas is not None else [x for x in DEFAULT_ALPHA if x<=a.max_alpha]
    result=scalar_sweep(z['base'],z['candidate'],z['target'],z['valid'],alphas,a.rmse_weight,a.max_alpha)
    result['predictions']=str(src.resolve())
    folder=Path(a.output) if a.output else src.parent/'alpha_sweep'
    save_sweep(result,folder)
    print('FARE_ALPHA_SWEEP '+json.dumps({'source':str(src),'best':result['best'],'output':str(folder)},ensure_ascii=False),flush=True)
    for row in result['grid']:
        print(f"FARE_ALPHA_ROW alpha={row['alpha']:.3f} MAE={row['MAE']:.6f} RMSE={row['RMSE']:.6f}",flush=True)
if __name__=='__main__':main()
