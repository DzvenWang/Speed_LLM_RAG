#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
import pandas as pd
import numpy as np

GADI_ROOT=next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd())
ROOT=GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baseline.utils import load_yaml
from baseline.shared_cache import cache_path, EVALUATION_SEVERITIES

CORE5=('entry_random','sensor_outage','temporal_outage','regional_outage','directional_outage')


def fmt(x):
    if x is None or (isinstance(x,float) and not np.isfinite(x)):
        return '-'
    try:
        return f'{float(x):.4f}'
    except Exception:
        return str(x)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--method',required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--formal-run-dir',default=None)
    a=ap.parse_args()
    cfg=load_yaml(ROOT/'configs/baseline.yaml')
    py=cfg['project']['python']
    exp=Path(cfg['project']['exp_root'])
    severity_rows=[]
    scenario_rows=[]
    for sev in [int(x) for x in cfg['experiment'].get('evaluation_severities',EVALUATION_SEVERITIES)]:
        print(f'========== EVALUATION s{sev} START method={a.method} seed={a.seed} ==========',flush=True)
        subprocess.check_call([py,str(ROOT/'scripts/preflight.py'),'--stage','evaluation','--method',a.method,'--severity',str(sev)])
        subprocess.check_call([py,str(ROOT/'scripts/preflight_formal.py'),'--method',a.method,'--seed',str(a.seed),'--stage','evaluation','--severity',str(sev)])
        cmd=[py,'-u',str(ROOT/'scripts/run_evaluation.py'),'--method',a.method,'--seed',str(a.seed),'--severity',str(sev)]
        if a.formal_run_dir: cmd += ['--formal-run-dir',str(a.formal_run_dir)]
        subprocess.check_call(cmd)
        rd=exp/'evaluation'/a.method/cfg['experiment']['subgraph']/f'seed{a.seed}'/f's{sev}'
        mf=rd/'test_metrics.csv'
        if not mf.exists():
            raise FileNotFoundError(mf)
        df=pd.read_csv(mf)
        got=[str(x) for x in df.loc[df.family.astype(str)!='ALL','family'].tolist()]
        if set(got)!=set(CORE5) or len(got)!=5:
            raise RuntimeError(f'SCENARIO_COVERAGE_FAILED s{sev}: expected={list(CORE5)} got={got}')
        print(f'SCENARIO_COVERAGE_PASS s{sev} families={got}',flush=True)
        allr=df.loc[df.family.astype(str)=='ALL'].iloc[0]
        severity_rows.append({
            'Severity':f's{sev}','MAE':allr['overall_MAE'],'RMSE':allr['overall_RMSE'],
            'Masked_MAE':allr['masked_MAE'],'Support_MAE':allr['support_MAE'],
            'H12_MAE':allr['H12_overall_MAE'],'H12_Support_MAE':allr['H12_support_MAE'],
        })
        cache=cache_path(cfg,'evaluation',sev)
        meta=pd.read_csv(cache/'test_manifest.csv')
        counts=meta.family.astype(str).value_counts().to_dict()
        for family in CORE5:
            r=df.loc[df.family.astype(str)==family].iloc[0]
            scenario_rows.append({
                'Severity':f's{sev}','Scenario':family,'Queries':int(counts.get(family,0)),
                'MAE':r['overall_MAE'],'RMSE':r['overall_RMSE'],'Masked_MAE':r['masked_MAE'],
                'Support_MAE':r['support_MAE'],'H12_MAE':r['H12_overall_MAE'],
                'H12_Support_MAE':r['H12_support_MAE'],
            })
        print(f'========== EVALUATION s{sev} PASS ==========',flush=True)
    out=exp/'evaluation'/a.method/cfg['experiment']['subgraph']/f'seed{a.seed}'
    sevdf=pd.DataFrame(severity_rows)
    scdf=pd.DataFrame(scenario_rows)
    sevdf.to_csv(out/'severity_summary.csv',index=False)
    scdf.to_csv(out/'scenario_summary.csv',index=False)

    print('\n'+'='*118)
    print('FINAL SEVERITY SUMMARY')
    print('='*118)
    show=sevdf.copy()
    for c in show.columns[1:]:
        show[c]=show[c].map(fmt)
    print(show.to_string(index=False))

    print('\n'+'='*148)
    print('FINAL SCENARIO SUMMARY')
    print('Scenario = Core-5 missing-sensing family; metrics are micro-aggregated within that scenario.')
    print('='*148)
    show=scdf.copy()
    for c in ['MAE','RMSE','Masked_MAE','Support_MAE','H12_MAE','H12_Support_MAE']:
        show[c]=show[c].map(fmt)
    print(show.to_string(index=False))
    print(f'\nSEVERITY_SUMMARY_CSV={out / "severity_summary.csv"}')
    print(f'SCENARIO_SUMMARY_CSV={out / "scenario_summary.csv"}')
    print(f'BASELINE MULTI-SEVERITY EVALUATION PASS method={a.method} seed={a.seed}')

if __name__=='__main__':
    main()
