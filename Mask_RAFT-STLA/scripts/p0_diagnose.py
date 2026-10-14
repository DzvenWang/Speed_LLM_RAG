#!/usr/bin/env python3
"""Paired P0 test evidence value diagnosis. Never retrains or changes a checkpoint."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
METHOD='Mask_RAFT-STLA'
from mask_raft_stla.config import cache_path, relocate_saved_config
from mask_raft_stla.p0 import (SCHEMA,read_predictions,validate_pair,sha256,
                            query_metrics,support_stratum,summary_table)

BASE_MODE='no_evidence'


def main():
    parser=argparse.ArgumentParser(description='P0 independent + paired retrieval diagnostics')
    parser.add_argument('--checkpoint',required=True,help='existing full v2.3 best.pt')
    parser.add_argument('--no-retrieval-checkpoint',help='separately TRAINED no_retrieval best.pt, optional')
    parser.add_argument('--severities',nargs='+',type=int,default=[20,40,60,90])
    parser.add_argument('--ensure-audit',action='store_true',help='run audit evaluator only for missing/stale predictions; requires inference runtime')
    parser.add_argument('--backbone-checkpoint',help='override relocated initial STLA checkpoint for evaluation')
    parser.add_argument('--bootstrap',type=int,default=400,help='query-level paired bootstrap, descriptive only')
    parser.add_argument('--output',help='optional P0 output directory (default <full_run>/p0_retrieval)')
    args=parser.parse_args()
    ck=Path(args.checkpoint).expanduser().resolve()
    ctl=Path(args.no_retrieval_checkpoint).expanduser().resolve() if args.no_retrieval_checkpoint else None
    full,ind=validate_pair(ck,ctl,METHOD)
    cfg=relocate_saved_config(full['config'])
    os.environ['TRAFFIDENT_TARGET']=cfg['experiment']['target']
    if args.ensure_audit:
        cmd=[sys.executable,'-u',str(ROOT/'scripts/evaluate_retrieval.py'),
             '--checkpoint',str(ck),'--reuse','--severities',*map(str,args.severities)]
        if ctl:cmd+=['--no-retrieval-checkpoint',str(ctl)]
        if args.backbone_checkpoint:cmd+=['--backbone-checkpoint',str(Path(args.backbone_checkpoint).resolve())]
        subprocess.check_call(cmd,cwd=ROOT)
    fp=sha256(ck);cp=sha256(ctl) if ctl else None
    all_rows=[]
    for sev in args.severities:
        full_prediction,base=read_predictions(ck,sev,'full',fp)
        wrong,_=read_predictions(ck,sev,'cross_history',fp)
        disabled,_=read_predictions(ck,sev,BASE_MODE,fp)
        independent=read_predictions(ctl,sev,'no_retrieval',cp)[0] if ctl else None
        test_root=cache_path(cfg,'evaluation',sev)
        y=np.load(test_root/'test_y_raw.npy',mmap_mode='r')
        valid=np.load(test_root/'test_future_valid.npy',mmap_mode='r')
        direct=np.load(test_root/'test_direct_mask.npy',mmap_mode='r')
        observed=np.load(test_root/'test_history_observed.npy',mmap_mode='r')
        meta=pd.read_csv(test_root/'test_manifest.csv')
        n=len(y)
        if len(meta)!=n or observed.shape[0]!=n or full_prediction.shape[0]!=n:
            raise RuntimeError(f's{sev}: predictions and test cache lack identical row order/length')
        frame=query_metrics(y,valid,full_prediction,base,wrong,disabled,independent,direct)
        frame.insert(0,'query_index',np.arange(n))
        frame.insert(0,'severity',sev)
        for col in ('window_start','family','view_group_id'):
            frame[col]=meta[col].to_numpy() if col in meta else ('NA' if col!='window_start' else np.arange(n))
        cover=np.asarray(observed).mean((1,2))
        frame['coverage']=cover
        frame['support_bin']=[support_stratum(x) for x in cover]
        frame['zero_history_node_fraction']=(np.asarray(observed).sum(1)==0).mean(1)
        frame['direct_missing_node_fraction']=np.asarray(direct).mean(1)
        frame['backbone_mode']=cfg['experiment'].get('backbone_mode','unknown')
        frame['method']=METHOD
        all_rows.append(frame)
        good=frame.gain_mae.to_numpy(dtype=float)
        print(f'P0_SEVERITY severity={sev} queries={n} beneficial_query_fraction={(good>0).mean():.4f} '
              f'harmful_query_fraction={(good<0).mean():.4f} '
              f'independent_control={ctl is not None}',flush=True)
    frame=pd.concat(all_rows,ignore_index=True)
    output=Path(args.output).resolve() if args.output else ck.parent/'p0_retrieval'
    output.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output/'p0_query_level.csv',index=False)
    groups=summary_table(frame)
    groups.to_csv(output/'p0_group_summary.csv',index=False)
    summary=[]
    for sev,part in frame.groupby('severity'):
        row={'severity':int(sev),'queries':len(part)}
        for key in ('full_mae','base_mae','cross_mae','disabled_mae','independent_mae',
                    'gain_mae','cross_history_gain','disabled_gain','independent_gain'):
            if key in part: row[key]=float(part[key].mean())
        row['beneficial_query_rate']=float((part.gain_mae>0).mean())
        row['harmful_query_rate']=float((part.gain_mae<0).mean())
        summary.append(row)
    summary=pd.DataFrame(summary)
    summary.to_csv(output/'p0_severity_summary.csv',index=False)
    # Unlike simple mean-of-query MAEs, this reproduces valid-element MAE on
    # a common paired support, to match the official metric protocol.
    element_rows=[]
    for sev,part in frame.groupby('severity'):
        counts=part['valid_count'].to_numpy(dtype=float)
        row={'severity':int(sev),'valid_elements':int(counts.sum())}
        for metric in ('full','base','cross','disabled','independent'):
            col=f'{metric}_mae'
            if col not in part:continue
            vals=part[col].to_numpy(dtype=float)
            ok=np.isfinite(vals)&(counts>0)
            row[col]=float(np.average(vals[ok],weights=counts[ok])) if ok.any() else float('nan')
        if 'independent_mae' in row:row['independent_gain_mae']=row['independent_mae']-row['full_mae']
        row['wrong_evidence_gain_mae']=row['cross_mae']-row['full_mae']
        row['disabled_gain_mae']=row['disabled_mae']-row['full_mae']
        element_rows.append(row)
    element_frame=pd.DataFrame(element_rows)
    element_frame.to_csv(output/'p0_element_weighted_summary.csv',index=False)
    bootstrap=[]
    if args.bootstrap:
        rng=np.random.default_rng(10)
        for sev,part in frame.groupby('severity'):
            for comparison in ['cross_history_gain','disabled_gain','independent_gain']:
                if comparison not in part:continue
                good=part[comparison].to_numpy(dtype=float)
                good=good[np.isfinite(good)]
                if not len(good):continue
                # Descriptive paired-query bootstrap, NOT a formal independent
                # traffic-window confidence interval (windows can overlap).
                means=[]
                for _ in range(args.bootstrap):
                    sample=rng.integers(0,len(good),len(good))
                    means.append(float(good[sample].mean()))
                bootstrap.append({'severity':int(sev),'comparison':comparison,
                                  'gain_mae':float(good.mean()),'ci_2.5':float(np.percentile(means,2.5)),
                                  'ci_97.5':float(np.percentile(means,97.5)),
                                  'note':'Descriptive query bootstrap; overlapping time windows are dependent'})
    pd.DataFrame(bootstrap).to_csv(output/'p0_descriptive_bootstrap.csv',index=False)
    report={'schema':SCHEMA,'method':METHOD,'dataset':cfg['experiment']['dataset'],
            'target':cfg['experiment']['target'],'seed':cfg['experiment']['seed'],
            'backbone_mode':cfg['experiment'].get('backbone_mode'),
            'full_checkpoint':str(ck),'full_checkpoint_sha256':fp,
            'backbone_sha256':full.get('backbone_sha256'),
            'independent_no_retrieval_checkpoint':str(ctl) if ctl else None,
            'independent_no_retrieval_sha256':cp,
            'full_vs_base':'Within-checkpoint internal base ablation, NOT independently trained baseline.',
            'full_vs_disabled':f'Test intervention {BASE_MODE}, not independently trained.',
            'full_vs_cross':'Wrong historical evidence intervention from same checkpoint.',
            'full_vs_independent':'Independent retrained no_retrieval control ONLY when supplied.',
            'oracle_note':'Run p0_candidate_probe.py for offline label-oracle candidate diagnostics; never a deployable metric.',
            'bootstrap_warning':'Query-resampling descriptive only; overlapping traffic windows violate independence.',
            'severities':args.severities,
            'avg_severity_equal_weight':{key:float(summary[key].mean()) for key in
                ('full_mae','base_mae','cross_mae','disabled_mae','independent_mae') if key in summary},
            'official_protocol_element_weighted_avg_severity':{key:float(element_frame[key].mean()) for key in ('full_mae','base_mae','cross_mae','disabled_mae','independent_mae') if key in element_frame},
            'outputs':['p0_query_level.csv','p0_group_summary.csv','p0_severity_summary.csv','p0_element_weighted_summary.csv','p0_descriptive_bootstrap.csv']}
    (output/'p0_manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('P0_PAIRED_PASS',output,flush=True)

if __name__=='__main__': main()
