"""FARE v5 fixed-checkpoint evidence-identity audit.

Audit metrics are descriptive; the Test split must not be used to repeatedly
select architecture/hyperparameters. Use validation for development decisions.
"""
from __future__ import annotations
import csv,json
from pathlib import Path
import torch
from .evaluation import evaluate_modes

MODES=('full','no_evidence','random_memory','shuffled_memory',
       'state_similarity_only','matched_random','candidate_prior','no_hidden_injection','no_forecast_correction','no_risk_control')

def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text('',encoding='utf8');return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)

@torch.no_grad()
def analyze(model,memory,datasets,device,cfg,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    from torch.utils.data import Subset
    rows=[]
    for split,ds in datasets.items():
        limit=int(cfg.get('analysis',{}).get('max_intervention_queries',512))
        if len(ds)>limit:ds=Subset(ds,list(range(limit)))
        result,_=evaluate_modes(model,memory,ds,device,cfg,output/split,list(MODES),save_traces=False)
        baseline=result['full']['overall_MAE']
        for mode,r in result.items():
            rows.append({'split':split,'mode':mode,'MAE':r.get('overall_MAE'),
                         'RMSE':r.get('overall_RMSE'),
                         'delta_MAE_vs_full':None if r.get('overall_MAE') is None else r['overall_MAE']-baseline,
                         'sampled_queries':len(ds),
                         'note':'fixed_checkpoint_exploratory_not_independently_retrained'})
    write_csv(output/'retrieval_identity_audit.csv',rows)
    (output/'scope.json').write_text(json.dumps({'scope':'fixed_checkpoint',
         'warning':'Test-split audit is descriptive; do not tune the next version using Test results',
         'modes':MODES},indent=2),encoding='utf8')
    print('FARE5_RETRIEVAL_AUDIT_COMPLETE',flush=True)
    return rows
