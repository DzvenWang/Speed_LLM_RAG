#!/usr/bin/env python3
"""Summarize canonical Base_SupportBench metrics for FARE Val and Test.

Across severities an equal-weight macro average is shown for comparison with
existing tables. Inside each severity, metrics are pooled over valid CELLS by
the shared experiment_common.metrics.metric_bundle API (never query averages).
"""
import argparse
import csv
import json
import math
from pathlib import Path


def _fmt_percent(value):
    return 'N/A' if value is None or not math.isfinite(float(value)) else f'{float(value):.2f}%'


def _macro(rows, field):
    vals=[r.get(field) for r in rows]
    return sum(float(v) for v in vals)/len(vals) if vals and all(v is not None and math.isfinite(float(v)) for v in vals) else None


def create_report(run_dir):
    run_dir=Path(run_dir)
    root=run_dir/'evaluation_candidate'
    if not root.exists():
        raise FileNotFoundError(root)
    files=sorted(root.glob('val_*_summary.json'))+sorted(root.glob('s*/s*_*_summary.json'))
    if not files:
        raise FileNotFoundError(f'No evaluation summaries in {root}')
    rows=[]
    for path in files:
        obj=json.loads(path.read_text(encoding='utf-8'))
        split='val' if path.parent==root else 'test'
        severity='training-mask validation' if split=='val' else path.parent.name
        final=obj['final'];base=obj['base'];cand=obj['candidate']
        if final.get('mape_protocol') != 'raw_target_percent_exclude_abs_le_1_v1':
            raise ValueError(f'Not evaluated using the shared Base_SupportBench MAPE protocol: {path}')
        rec={
            'split':split,'severity':severity,'rows':int(obj.get('rows',0)),
            'metrics_revision':final['metrics_revision'],
            'mape_protocol':final['mape_protocol'],
            'mape_min_target_abs':final['mape_min_target_abs'],
        }
        for method,metrics in [('base',base),('candidate',cand),('final',final)]:
            for scope,pre in [('', ''), ('directly_masked_', 'directly_masked_')]:
                for metric in ('MAE','RMSE','MAPE','count','MAPE_count','MAPE_coverage'):
                    rec[f'{method}_{scope}{metric}']=metrics.get(pre+metric)
            for h in (3,6,12):
                for metric in ('MAE','RMSE','MAPE','MAPE_count','MAPE_coverage'):
                    rec[f'{method}_H{h}_{metric}']=metrics.get(f'H{h}_{metric}')
                    rec[f'{method}_H{h}_masked_{metric}']=metrics.get(f'H{h}_masked_{metric}')
        denom=float(base['MAE'])
        rec['gain_MAE_pct']=100*(denom-float(final['MAE']))/denom if denom and math.isfinite(denom) else float('nan')
        rows.append(rec)
    # All rows have the same schema; canonical MAPE fields remain named as in the bundle.
    csv_file=run_dir/'pilot_val_test_metrics.csv'
    with csv_file.open('w',newline='',encoding='utf-8') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (run_dir/'pilot_val_test_metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    lines=[
        '# FARE Pilot: Base_SupportBench Shared-Metric Validation and Test','',
        '**Checkpoint selected exclusively by Validation MAE. Test results are exploratory.**','',
        f'MAPE protocol: `{rows[0]["mape_protocol"]}`; raw-target |truth| > {rows[0]["mape_min_target_abs"]}. '
        'Both Speed and Flow use the same strict threshold. MAE/RMSE masks are unchanged.','',
        '| Split | Severity | Base MAE | FARE MAE | MAE Gain | Base RMSE | FARE RMSE | Base MAPE | FARE MAPE | MAPE count | MAPE coverage |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        lines.append(
            f'| {r["split"]} | {r["severity"]} | {r["base_MAE"]:.4f} | {r["final_MAE"]:.4f} | '
            f'{r["gain_MAE_pct"]:+.2f}% | {r["base_RMSE"]:.4f} | {r["final_RMSE"]:.4f} | '
            f'{_fmt_percent(r["base_MAPE"])} | {_fmt_percent(r["final_MAPE"])} | '
            f'{r["final_MAPE_count"]} | {r["final_MAPE_coverage"]:.1%} |'
        )
    test=[r for r in rows if r['split']=='test']
    if test:
        lines.extend(['','### Test macro average (equal weight across the evaluated severities)',''])
        avgs={k:_macro(test,k) for k in ('base_MAE','candidate_MAE','final_MAE',
                                           'base_RMSE','candidate_RMSE','final_RMSE',
                                           'base_MAPE','candidate_MAPE','final_MAPE')}
        gain=100*(avgs['base_MAE']-avgs['final_MAE'])/avgs['base_MAE'] if avgs['base_MAE'] else float('nan')
        lines.append(
            f'Base MAE: **{avgs["base_MAE"]:.4f}**; FARE MAE: **{avgs["final_MAE"]:.4f}**; '
            f'MAE gain: **{gain:+.2f}%**; Base RMSE: **{avgs["base_RMSE"]:.4f}**; '
            f'FARE RMSE: **{avgs["final_RMSE"]:.4f}**; '
            f'Base MAPE: **{_fmt_percent(avgs["base_MAPE"])}**; '
            f'FARE MAPE: **{_fmt_percent(avgs["final_MAPE"])}**.'
        )
        lines.extend(['', 'The above severity macro averages are *not* pooled Test metrics. '
                      'Within each split/severity, shared metrics are micro-aggregated across eligible cells. '
                      'MAPE excludes |truth| <= 1.0, and its coverage is shown.'])
    md=run_dir/'pilot_val_test_metrics.md'
    md.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return md,rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    args=parser.parse_args()
    path,rows=create_report(args.run_dir)
    print('FARE_PILOT_VAL_TEST_REPORT '+json.dumps({'output':str(path),
          'splits':sorted({r['split'] for r in rows}),'rows':len(rows)}),flush=True)

if __name__=='__main__':
    main()
