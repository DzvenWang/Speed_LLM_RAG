"""FARE v2.1.3 Oracle Audit reports: no test-time model selection."""
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
    md=['# FARE v2.1.3 Candidate-Level Oracle Audit','',
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


COUNTERFACTUAL_FIELDS=(
    'real_gain_vs_region_null',
    'loo_mean_keep_gain','loo_harmful_slot_fraction',
    'loo_best_removal_improvement','node_loo_mean_keep_gain',
    'node_loo_harmful_fraction',
    'replacement_best_improvement_over_real',
    'replacement_random_mean_gain_vs_real',
    'replacement_beneficial_fraction','replacement_ranker_spearman',
    'best_scaling_improvement_over_real',
    'best_scaling_alpha',
    'loo_direct_mean_keep_gain',
    'replacement_direct_mean_gain',
    'node_loo_direct_mean_keep_gain',
    *tuple(f'loo_horizon_group_{i}_mean_keep_gain' for i in (1,2,3)),
    *tuple(f'replacement_horizon_group_{i}_mean_gain' for i in (1,2,3)),
    *tuple(f'node_loo_horizon_group_{i}_mean_keep_gain' for i in (1,2,3)),
)


def summarize_counterfactual(rows, output: Path):
    """Separate 2.1.3 summary from backward-compatible singleton reports."""
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    grouped=defaultdict(list)
    for row in rows:
        if 'counterfactual_scope' in row:
            grouped[(row['split'],row.get('severity'))].append(row)
    report=[]
    for (split,severity),items in sorted(grouped.items(),key=lambda kv:(kv[0][0],-1 if kv[0][1] is None else kv[0][1])):
        item={'split':split,'severity':severity,
              'query_regions':len(items),
              'queries':len({row['row_id'] for row in items})}
        for field in COUNTERFACTUAL_FIELDS:
            item[field]=finite_mean([row.get(field) for row in items])
        item['fraction_regions_with_harmful_slot']=finite_mean([
            any(x['keep_gain_vs_removing']<0 for x in row.get('leave_one_out',[]))
            for row in items if row.get('leave_one_out')])
        item['fraction_regions_where_scaling_better_than_real']=finite_mean([
            row['best_scaling_improvement_over_real']>1e-6
            for row in items if row.get('correction_scaling')])
        item['mean_loo_slots_evaluated']=finite_mean([len(row.get('leave_one_out',[])) for row in items])
        item['mean_replacements_evaluated']=finite_mean([len(row.get('same_weight_replacements',[])) for row in items])
        item['mean_node_loo_evaluated']=finite_mean([len(row.get('node_leave_one_out',[])) for row in items])
        report.append(item)
    (output/'counterfactual_summary.json').write_text(json.dumps(clean(report),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    if report:
        with (output/'counterfactual_summary.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(report[0]))
            writer.writeheader();writer.writerows(clean(report))
    md=['# FARE v2.1.3 Weight-Preserving Counterfactual Audit','',
        '**Fixed checkpoint. Hindsight comparisons, not deployed policies. Test is descriptive only.**','',
        'MAE gains use original target units; positive means better. Other regions retain the trained Real mixture.','',
        '- LOO removes one **per-node rank slot** across the target region, transferring exactly its probability mass to Null. Positive keep-gain means retaining the slot is helpful.',
        '- Same-weight replacement substitutes a causal Top-64 candidate in one node-specific rank slot. The original slot weight remains unchanged wherever that candidate is valid; otherwise the slot mass goes to Null.',
        '- Correction scaling interpolates **final predictions**: `Null + alpha*(Real-Null)`, a retrospective prediction-space region counterfactual, not a trained controller or a structural reinjection.',
        '- Node LOO removes one slot from one node, preserving the other node mixtures. Horizon gains are MAE differences for horizon thirds.',
        '- Best replacement / best scaling is selected using evaluation labels and is **not** an inference-time upper bound on arbitrary mixtures.','',
        '| Split | Severity | Query×Region | Real Gain vs Null | Mean Slot Keep Gain | Harmful Slots | Best LOO Removal | Best Same-Weight Replacement | Replacement Random Mean | Replacement Spearman | Best Alpha | Best Scaling Improvement |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(value):
        return '—' if value is None else f'{value:.5f}'
    for x in report:
        md.append('| '+' | '.join([
            str(x['split']),str(x['severity'] if x['severity'] is not None else 'val'),
            str(x['query_regions']),fmt(x['real_gain_vs_region_null']),
            fmt(x['loo_mean_keep_gain']),fmt(x['loo_harmful_slot_fraction']),
            fmt(x['loo_best_removal_improvement']),
            fmt(x['replacement_best_improvement_over_real']),
            fmt(x['replacement_random_mean_gain_vs_real']),
            fmt(x['replacement_ranker_spearman']),fmt(x['best_scaling_alpha']),
            fmt(x['best_scaling_improvement_over_real'])])+' |')
    md.extend(['','## How to use the report','',
        '- **Many harmful LOO slots:** examine learned weight allocation and Null control before expanding Memory or Top-K.',
        '- **Replacement Oracle high, ranker correlation low:** investigate query-conditioned utility supervision on the *same-weight* counterfactual target.',
        '- **Scaling alpha below 1 often wins:** inspect correction magnitude, gate calibration and Speed s90.',
        '- **Region LOO improves but node LOO differs:** candidate usefulness is node-specific, not a uniform region property.',
        '- The two audits have different intervention semantics; do not subtract singleton Oracle from same-weight replacement gains.','',
        'Small fixed samples and overlapping time windows do not establish statistical significance.',
        'Oracle and hindsight Alpha must never be chosen using Test for deployed model decisions.'])
    (output/'counterfactual_summary.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    return report
