"""Consolidated report; recommendations are hypotheses, not automated paper claims."""
from __future__ import annotations
from pathlib import Path
import csv,json,datetime
import numpy as np
from .engine import summarize

def save_json(path,data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')

def format_v(v):
    if v is None:return 'N/A'
    if isinstance(v,bool):return 'PASS' if v else 'FAIL'
    if isinstance(v,float):return f'{v:.5f}'
    return str(v)

def report(rows, output, provenance, expected=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    summary=summarize(rows)
    coverage=[]
    expected=expected or {}
    targets=expected.get('targets',['speed','flow'])
    severities=expected.get('severities',[20,40,60,90])
    families=expected.get('families',[])
    got={(z['target'],z['severity'],z['failure_family']) for z in rows}
    if families:
        coverage=[(t,int(s),f) for t in targets for s in severities for f in families if (t,int(s),f) not in got]
    # Each shard is a validation stratum; macro aggregation is equal shard weighting.
    decisions=[]
    for t,x in summary.items():
        r=[z for z in rows if z['target']==t]
        if any(z.get('null_path_equivalent_1e-5') is False for z in r):
            decisions.append({'target':t,'priority':'P0','module':'null_gate','finding':'Null evidence does not reproduce same-checkpoint no-evidence forecast','action':'Trace hidden and forecast injection; require global rejection path'})
        good=[z for z in r if z.get('oracle_best_candidate_gain') is not None]
        if not good:
            decisions.append({'target':t,'priority':'BLOCKER','module':'retrieval_oracle','finding':'No selected-slot candidate predictions exported','action':'Export pred_candidates from same frozen checkpoint; no claim about oracle availability'})
        else:
            pot=float(np.mean([z['oracle_best_candidate_gain'] for z in good]))
            realize=float(np.mean([z['realized_gain_vs_oracle'] for z in good]))
            if pot>1e-8 and realize/pot <0.3:
                decisions.append({'target':t,'priority':'P0','module':'retrieval_vs_transfer','finding':f'Low realized/selected-slot-oracle ratio (macro {realize/pot:.3f})','action':'Inspect candidate selection and evidence integration separately; selected-slot oracle is a limited diagnostic, not the whole retrieval-pool upper bound'})
        if all('similarity_only_gain_query_mean' not in z for z in r):
            decisions.append({'target':t,'priority':'P1','module':'similarity_baseline','finding':'Similarity-only comparison not exported','action':'Do not claim advantage over similarity retrieval'})
        if all('hidden_only_gain_query_mean' not in z for z in r):
            decisions.append({'target':t,'priority':'P1','module':'injection','finding':'Hidden-only comparison missing','action':'Do not assign causal credit to hidden injection'})
        if all('no_gate_gain_query_mean' not in z for z in r):
            decisions.append({'target':t,'priority':'P1','module':'risk','finding':'Gate ablation missing','action':'No verified evidence that risk gate reduces harm'})
    if coverage:
        decisions.insert(0,{'target':'both','priority':'BLOCKER','module':'coverage','finding':f'{len(coverage)} missing target/severity/family shards','action':'Do not interpret as a complete one-pass experiment'})
    doc={'generated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'experiment':'FARE v4.0.4 one-pass observational/paired counterfactual diagnostic',
         'provenance':provenance,'coverage_missing':coverage,'aggregate':summary,
         'decisions':decisions,'strata':rows}
    save_json(output/'diagnostic_report.json',doc)
    save_json(output/'decisions.json',decisions)
    with (output/'overall_comparison.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['target','severity','family','queries','full_MAE','full_RMSE','no_evidence_MAE','no_evidence_RMSE','gain_query_mean','harm_rate','oracle_best_candidate_gain','oracle_capture_ratio','null_equivalent'])
        for z in sorted(rows,key=lambda q:(q['target'],q['severity'],q['failure_family'])):
            w.writerow([z['target'],z['severity'],z['failure_family'],z['queries'],
                        z['modes']['full']['MAE'],z['modes']['full']['RMSE'],
                        z['modes']['no_evidence']['MAE'],z['modes']['no_evidence']['RMSE'],
                        z['gain_MAE_query_mean'],z['harm_rate'],z.get('oracle_best_candidate_gain'),z.get('oracle_capture_ratio'),z.get('null_path_equivalent_1e-5')])
    with (output/'module_contribution.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['target','severity','family','mode','MAE','RMSE','difference_vs_full_MAE'])
        for z in rows:
            for mode,met in z['modes'].items():
                w.writerow([z['target'],z['severity'],z['failure_family'],mode,met['MAE'],met['RMSE'],met['MAE']-z['modes']['full']['MAE']])
    lines=['# FARE v4.0.4 — One-pass Diagnostic Report','',
           f'**Source split: {provenance.get("source_split","unknown")}**. Test-split diagnostics are exploratory and MUST NOT be represented as independent validation. All interventions share frozen weights.','',
           '## Dataset coverage','',f'Analyzed {len(rows)} {provenance.get("source_split","unknown")} shards. Missing expected shards: **{len(coverage)}**.','',
           '## Overall','', '|Target|Shards|Queries|Full MAE (macro)|No-evidence MAE (macro)|Harm rate (macro)|Oracle capture (macro)|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for t,x in summary.items():
        lines.append('|'+ '|'.join([t,str(x['shards']),str(x['queries']),format_v(x['mean_shard_full_MAE']),format_v(x['mean_shard_no_evidence_MAE']),format_v(x['mean_shard_harm_rate']),format_v(x['mean_oracle_capture_ratio'])])+'|')
    lines+=['','## Next-version decisions','']
    for d in decisions:lines.append(f"- **{d['priority']} · {d['module']} · {d['target']}:** {d['finding']}. {d['action']}.")
    lines+=['','## Interpretational restrictions','',
            '- This compares inference interventions, **not** independently re-trained model ablations.',
            '- Oracle uses source-split labels post hoc and MUST NOT be deployed for prediction. Selected-slot oracle is NOT a full retrieval-pool upper bound.',
            '- Pooled MAE/RMSE and macro-per-stratum MAE are different quantities; this report labels macro values.',
            '- Query-cluster bootstrap intervals are descriptive; spatial-temporal dependence can make them optimistic.',
            '- Checkpoint-in-sample residual memory remains exploratory until OOF/time-availability checks pass.',
            '- Missing diagnostic modes are explicitly reported; never fill missing values with zero or inferred gains.',
            '']
    (output/'diagnostic_report.md').write_text('\n'.join(lines),encoding='utf-8')
    return doc
