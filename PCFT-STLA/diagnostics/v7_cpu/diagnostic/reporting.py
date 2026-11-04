"""Small reproducible outputs; no checkpoint modifications."""
from __future__ import annotations
import csv,json
from pathlib import Path


def write_csv(path,rows,columns=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    rows=list(rows)
    fields=list(columns or [])
    if not fields:
        for row in rows:
            for key in row:
                if key not in fields:fields.append(key)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def json_write(path,document):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(document,ensure_ascii=False,indent=2,sort_keys=True,default=str)+'\n',encoding='utf-8')


def generate_report(out,summary,groups,interventions):
    main=summary.get('global',{})
    lines=['# PCFT v7 CPU Diagnostic Report','',
           'This is a **post-training, read-only, paired observational/intervention analysis**.',
           'Post-training interventions are **not retrained ablations**. Oracle predictions use truth only for diagnosis.',
           'C1 does not output a calibrated imputed historical state; C1 here measures coverage/support behavior.',
           '',f"- Target: **{summary['target']}**",f"- Split: **{summary['split']}**",
           f"- Windows: **{summary['windows']}**",f"- Valid forecast positions: **{main.get('valid_positions',0)}**",'']
    for k in ('base_mae','local_mae','propagation_mae','controlled_mae','uniform_mae','fixed_shrink_mae','hard_action_mae','oracle_discrete_mae','overall_improvement_pct','propagation_vs_local_pct'):
        if k in main:lines.append(f'- {k}: **{main[k]:.6f}**')
    lines.extend(['','## Interpretation priorities',
       '- Compare controlled_mae to uniform and fixed_shrink on identical windows. If similar, C3 needs stronger conditional control.',
       '- Inspect target_coverage, last_observation_age and longest_missing_streak groups to assess whether C1 should retain temporal order.',
       '- Inspect available_incoming_edges and weighted_source_age to check whether C2 lacks usable/fresh sources.',
       '- Inspect benefit and harm in the same groups; mean edge/attention weight is not forecast gain.',
       '- Treat source-removal and coverage interventions as behavior tests, not causal road-propagation estimates.',
       '','## Interventions (paired frozen model)'])
    for r in interventions:
        if r.get('strategy')=='recomputed_policy':
            lines.append(f"- {r.get('intervention')}: MAE={r.get('mae','NA')}, delta_vs_original={r.get('delta_vs_original','NA')}, valid={r.get('valid_positions')}")
    lines.extend(['','## Artifact files',
        '- summary.json, run_manifest.json, sample_manifest.csv',
        '- c1_support_groups.csv, c2_source_groups.csv, c2_message_statistics.csv, c3_action_statistics.csv, c3_gain_calibration.csv',
        '- horizon_results.csv, intervention_results.csv, selected_cases.csv',
        '- window_metrics.csv, bootstrap_intervals.json (time-block resampling)',
        '- Optional edge_weight_summary.csv (deep mode only)',
        '',f'- Number of stratified rows: {len(groups)}',''])
    (Path(out)/'report.md').write_text('\n'.join(lines),encoding='utf-8')
