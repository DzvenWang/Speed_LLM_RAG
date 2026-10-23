#!/usr/bin/env python3
"""Summarize independently trained M0/M1/M2/M3; refuse protocol mismatches.

Only reads finished results. Does not fit models or choose gains from Test.
Raises if target/seed/backbone/dataset, memory budget or protocol mismatches.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import pandas as pd
import torch

METHOD='PCFT'

def summarize(experiment_root:Path, dataset:str, output:Path, split='validation'):
    root=experiment_root/dataset/'formal'
    rows=[];metadata={};ref=None
    for d in sorted(root.glob('*')):
        ckpath=d/'best.pt';sp=d/'summary.json'
        if not (ckpath.exists() and sp.exists()):continue
        ck=torch.load(ckpath,map_location='cpu',weights_only=False)
        cfg=ck.get('config',{})
        if ck.get('version')!='3.0.0':continue
        level=str(cfg.get('v3',{}).get('contribution',''))
        if level not in ('M0','M1','M2','M3'):continue
        invariant={
            'backbone_sha256':ck['backbone_sha256'],
            'dataset':cfg['experiment']['dataset'],
            'target':cfg['experiment']['target'],
            'seed':cfg['experiment']['seed'],
            'backbone_mode':cfg['experiment'].get('backbone_mode'),
            'history':cfg['experiment']['history'],
            'horizon':cfg['experiment']['horizon'],
            'retrieval':cfg['retrieval'],
            'batch_size':cfg['training']['batch_size'],
            'max_epochs':cfg['training']['max_epochs'],
            'schedule':cfg['training']['schedule'],
            'lr_new':cfg['training']['lr_new'],
        }
        if ref is None:ref=invariant
        elif invariant!=ref:raise RuntimeError(f'Attribution protocol mismatch: {d}')
        if level in metadata:raise RuntimeError(f'Multiple {level} runs under {root}; isolate/select one run per level')
        summary=json.loads(sp.read_text())
        if split=='test':
            if not summary.get('test_evaluated'):raise RuntimeError(f'Missing independent test evaluation {d}')
            for row in summary.get('evaluation_summary',[]):
                rows.append({'level':level,'severity':int(row['severity']),
                             'MAE':float(row['MAE']),'RMSE':float(row['RMSE'])})
        else:
            val=summary['validation'];rows.append({'level':level,'severity':'validation',
                'MAE':float(val['MAE']),'RMSE':float(val['RMSE'])})
        metadata[level]={'run':str(d),'best_epoch':ck['best_epoch'],
                         'best_val_mae':ck['best_val_mae']}
    if set(metadata)!=set(('M0','M1','M2','M3')):
        raise RuntimeError(f'Independent attribution requires exactly M0/M1/M2/M3: found {sorted(metadata)}')
    df=pd.DataFrame(rows)
    target=df.pivot(index='severity',columns='level',values='MAE')
    for a,b,label in [('M0','M1','C1'),('M1','M2','C2'),('M2','M3','C3')]:
        target[label+'_MAE_gain']=target[a]-target[b]
    output.mkdir(parents=True,exist_ok=True)
    df.to_csv(output/'independent_models.csv',index=False)
    target.to_csv(output/'contribution_gains.csv')
    (output/'provenance.json').write_text(json.dumps({'method':METHOD,'split':split,
        'protocol':ref,'runs':metadata,'note':'M0-M3 differences are conditional, not causal effects.'},
        indent=2,default=str))
    print('ATTRIBUTION_PASS',METHOD,'output=',output)
    print(target.to_string(float_format=lambda x:f'{x:.6f}'))
    return target

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--experiment-root',required=True)
    ap.add_argument('--dataset',default='TraffiDent_Net_06')
    ap.add_argument('--output',required=True);ap.add_argument('--split',choices=['validation','test'],default='validation')
    a=ap.parse_args()
    summarize(Path(a.experiment_root),a.dataset,Path(a.output),a.split)
