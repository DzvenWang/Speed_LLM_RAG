"""FARE v2.1.2 Oracle Audit reports: no test-time model selection."""
from __future__ import annotations
import csv,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def finite_mean(values):
    vals=np.asarray([x for x in values if x is not None and math.isfinite(float(x))],dtype=float)
    return float(vals.mean()) if len(vals) else None


FIELDS=('real_gain_vs_null','oracle_best_singleton_gain','oracle_with_abstention_gain',
        'selected_top1_gain_vs_null','selected_union_best_singleton_gain',
        'random_singleton_expected_gain','ranker_predicted_top1_gain',
        'ranker_candidate_spearman','singleton_oracle_minus_ranker','positive_candidate_fraction',
        'harmful_candidate_fraction','direct_real_gain_vs_null',
        'real_mae','null_mae','base_mae','num_legal_coarse_candidates',
        'num_evaluated_candidates')


def summarize(rows,output:Path):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    grouped=defaultdict(list)
    for row in rows:grouped[(row['split'],row.get('severity'))].append(row)
    report=[]
    for (split,severity),items in sorted(grouped.items(),key=lambda kv:(kv[0][0],-1 if kv[0][1] is None else kv[0][1])):
        record={'split':split,'severity':severity,'sampled_query_regions':len(items),
                'sampled_queries':len(set(x['row_id'] for x in items))}
        for field in FIELDS:record[field]=finite_mean([x.get(field) for x in items])
        if record['oracle_with_abstention_gain'] is not None and record['real_gain_vs_null'] is not None:
            # NOT a strict upper bound on blended Real Top8 due to evidence synergy.
            record['singleton_oracle_minus_real']=record['oracle_with_abstention_gain']-record['real_gain_vs_null']
        else:record['singleton_oracle_minus_real']=None
        report.append(record)
    (output/'oracle_summary.json').write_text(json.dumps(clean(report),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    if report:
        with (output/'oracle_summary.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(report[0]));writer.writeheader();writer.writerows(clean(report))
    md=['# FARE v2.1.2 Candidate-Level Oracle Audit','',
        'This is a fixed-checkpoint **exploratory diagnostic**; no test labels enter inference.','',
        'The oracle selects the best **evaluated singleton** of one region while other regions retain Real evidence. It is **not** a globally joint Oracle@64 or guaranteed bound on the Real Top8 mixture.','',
        'All gains are **MAE(null target-region evidence) − MAE(candidate or Real)** in original target units. Positive is better.','',
        '| Split | Severity | Query×Region | Real Gain | Singleton Oracle + Null | Ranker Predicted Top1 | Spearman | Random Singleton | Selected Top1 | Positive Candidates | Direct Masked Real Gain |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x):return '—' if x is None else f'{x:.5f}'
    for x in report:
        md.append('| '+ ' | '.join([str(x['split']),str(x['severity'] if x['severity'] is not None else 'val'),
            str(x['sampled_query_regions']),fmt(x['real_gain_vs_null']),fmt(x['oracle_with_abstention_gain']),
            fmt(x['ranker_predicted_top1_gain']),fmt(x['ranker_candidate_spearman']),
            fmt(x['random_singleton_expected_gain']),fmt(x['selected_top1_gain_vs_null']),
            fmt(x['positive_candidate_fraction']),fmt(x['direct_real_gain_vs_null'])])+' |')
    md.extend(['','Interpretation:',
        '- Oracle higher than Real can indicate ranking and/or evidence-fusion limitations.',
        '- Oracle near zero can indicate weak candidate transport, weak recall, or an overly restrictive singleton test.',
        '- Compare Real with Random, Shuffled and matched retrained no-memory baselines before attributing gains to retrieval.',
        '- Sampled query/region counts are not a statistically representative full-Test estimate.',
        '- Test is diagnostic only; do not select checkpoints or tune hyperparameters on Test.'])
    (output/'oracle_summary.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    return report
