#!/usr/bin/env python3
"""Paired frozen-checkpoint v4 mechanism ablations and causal attribution.

Does NOT claim independent contributions of a retrained model; for decisive
ablation train each variant separately with train_v4.py --ablation MODE.
"""
from __future__ import annotations
import argparse,sys,subprocess
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=['full','no_pair_interaction','shuffled_cf_pairing','no_failure_residual',
         'no_normal_residual','no_support_conditioning','no_utility_ranking',
         'no_gain_harm','no_risk','similarity_only','no_retrieval']

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--modes',nargs='+',default=DEFAULT)
    p.add_argument('--severities',type=int,nargs='+',default=[20,40,60,90])
    p.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args()
    folder=Path(a.checkpoint).resolve().parent/'evaluation'
    rows=[]
    for mode in a.modes:
        cmd=[sys.executable,'-u',str(ROOT/'scripts/evaluate_v4.py'),
             '--checkpoint',a.checkpoint,'--evidence-mode',mode,
             '--batch-size',str(a.batch_size),'--severities',*[str(v) for v in a.severities]]
        subprocess.check_call(cmd,cwd=ROOT)
        table=pd.read_csv(folder/f'severity_summary.{mode}.csv')
        rows.extend(table.to_dict('records'))
    result=pd.DataFrame(rows)
    result.to_csv(folder/'mechanism_v4_comparison.csv',index=False)
    # All rows were evaluated on the same severity, query and backbone.
    if 'MAE' in result.columns:
        full=result[result['mode']=='full'].set_index('severity')['MAE']
        result['delta_mae_vs_full']=result.apply(
            lambda row:float(row['MAE']-full.loc[row['severity']]) if row['severity'] in full.index else float('nan'),axis=1)
        result.to_csv(folder/'mechanism_v4_comparison.csv',index=False)
    print('V4_DIAGNOSTIC_PASS',folder/'mechanism_v4_comparison.csv',flush=True)
if __name__=='__main__':main()
