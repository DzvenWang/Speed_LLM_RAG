#!/usr/bin/env python3
"""Summarize saved FARE val and per-severity Test metrics without model selection."""
import argparse,csv,json
from pathlib import Path


def create_report(run_dir):
    root=Path(run_dir)/'evaluation_candidate'
    if not root.exists():raise FileNotFoundError(root)
    files=sorted(root.glob('val_*_summary.json'))+sorted(root.glob('s*/s*_*_summary.json'))
    if not files:raise FileNotFoundError(f'No evaluation summaries in {root}')
    rows=[]
    for path in files:
        obj=json.loads(path.read_text(encoding='utf-8'))
        split='val' if path.parent==root else 'test'
        severity='training-mask validation' if split=='val' else path.parent.name
        final=obj['final'];base=obj['base'];cand=obj['candidate']
        value=float(base['MAE']);result=float(final['MAE'])
        rows.append({'split':split,'severity':severity,'rows':int(obj.get('rows',0)),
                     'base_MAE':value,'candidate_MAE':float(cand['MAE']),'final_MAE':result,
                     'gain_MAE_pct':100*(value-result)/value if value else 0,
                     'base_RMSE':float(base['RMSE']),'candidate_RMSE':float(cand['RMSE']),
                     'final_RMSE':float(final['RMSE'])})
    csv_file=Path(run_dir)/'pilot_val_test_metrics.csv'
    with csv_file.open('w',newline='',encoding='utf-8') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (Path(run_dir)/'pilot_val_test_metrics.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    lines=['# Pilot Prediction Evaluation: Validation and Test','',
           '**Validation selected the checkpoint; Test is exploratory only and must not be used for tuning.**','',
           '| Split | Severity | Base MAE | Candidate MAE | FARE MAE | MAE Gain | Base RMSE | FARE RMSE |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["split"]} | {r["severity"]} | {r["base_MAE"]:.4f} | '
                     f'{r["candidate_MAE"]:.4f} | {r["final_MAE"]:.4f} | '
                     f'{r["gain_MAE_pct"]:+.2f}% | {r["base_RMSE"]:.4f} | {r["final_RMSE"]:.4f} |')
    test=[r for r in rows if r['split']=='test']
    if test:
        lines+=['','### Test macro average (equal weight per severity)','']
        averages={k:sum(r[k] for r in test)/len(test) for k in ('base_MAE','candidate_MAE','final_MAE','base_RMSE','final_RMSE')}
        gain=100*(averages['base_MAE']-averages['final_MAE'])/averages['base_MAE'] if averages['base_MAE'] else 0
        lines += [f'Base MAE: **{averages["base_MAE"]:.4f}**; FARE MAE: **{averages["final_MAE"]:.4f}**; '
                  f'MAE improvement: **{gain:+.2f}%**; '
                  f'Base RMSE: **{averages["base_RMSE"]:.4f}**; FARE RMSE: **{averages["final_RMSE"]:.4f}**.',
                  '', 'Macro RMSE is the average of per-severity RMSE, not a pooled RMSE.']
    md=Path(run_dir)/'pilot_val_test_metrics.md'
    md.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return md,rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True)
    args=p.parse_args()
    path,rows=create_report(args.run_dir)
    print('FARE_PILOT_VAL_TEST_REPORT '+json.dumps({'output':str(path),
          'splits':sorted({r['split'] for r in rows}),'rows':len(rows)}),flush=True)


if __name__=='__main__':main()
