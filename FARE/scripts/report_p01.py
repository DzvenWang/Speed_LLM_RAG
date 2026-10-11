#!/usr/bin/env python3
"""Combine Speed/Flow P0.1 Validation-only diagnostic summaries."""
import argparse,json,csv
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root);reports={};rows=[]
    for target in ('speed','flow'):
        src=root/target/'p01_report.json'
        if not src.is_file():raise FileNotFoundError(f'Missing target P0.1 report {src}')
        j=json.loads(src.read_text(encoding='utf-8'))
        if j.get('status')!='PASS':raise ValueError(f'Incomplete {target}: {j}')
        reports[target]=j
        rows += j['full_per_severity']
    out={'status':'PASS','targets':reports,
         'warning':'All results are validation diagnostic remasks, not official held-out Test. Oracle probes access validation labels only in offline analysis.'}
    (root/'combined_report.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (root/'combined_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    lines=['# FARE P0.1 — Speed / Flow', '',
          'Validation only, canonically regenerated severity masks; not official Test.', '',
          '|Target|Severity|Base MAE|Candidate MAE|Final MAE|MAE gain %|',
          '|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['target']}|s{r['severity']}|{r['base_MAE']:.4f}|{r['candidate_MAE']:.4f}|{r['full_MAE']:.4f}|{r['MAE_gain_pct']:.3f}|")
    lines.extend(['','Interpretation must use `integrity_sensitivity.csv`, `candidate_utility.json`, `alpha_sweep.csv` and family diagnostics for each severity.',
                  'Oracle results are privileged validation diagnostics and must NOT be reported as achievable model performance.'])
    (root/'combined_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'FARE_P01_COMPLETE output={root} target_count=2 severity_rows={len(rows)}',flush=True)
if __name__=='__main__':main()
