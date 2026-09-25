#!/usr/bin/env python3
"""Summarize an already-completed explicit-checkpoint evaluation without rerunning inference."""
from pathlib import Path
import argparse
import datetime
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backbone.utils import load_yaml
from backbone.evaluation_reporting import (
    severity_row,
    scenario_rows,
    severity_table_lines,
    scenario_table_lines,
)


def _pick_summary(root: Path, seed: int, run_stamp: str | None) -> Path:
    if run_stamp:
        p = root / f'{run_stamp}_seed{seed}' / 'test_summary.json'
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    fs = list(root.glob(f'*_seed{seed}/test_summary.json')) if root.exists() else []
    if not fs:
        raise FileNotFoundError(f'No checkpoint evaluation summary under {root} for seed={seed}')
    return max(fs, key=lambda p: p.stat().st_mtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint-run-tag', required=True,
                    help='Formal run directory name, e.g. 20260913_150736_seed10')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--run-stamp', default=None,
                    help='Evaluation run stamp, e.g. 20260925_173920. Default: latest per severity.')
    ap.add_argument('--severities', nargs='+', type=int, default=[20,40,60,90], choices=[20,40,60,90])
    ap.add_argument('--config', default=str(ROOT / 'configs' / 'backbone.yaml'))
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    exp = Path(cfg['project']['exp_root'])
    base = exp / 'PatchSTG' / cfg['experiment']['subgraph'] / 'checkpoint_evaluation' / args.checkpoint_run_tag
    families = [str(x) for x in cfg['experiment']['families']]
    severity_rows = []
    scenario_all = []
    source_dirs = []

    for sev in args.severities:
        summary_path = _pick_summary(base / f's{sev}', args.seed, args.run_stamp)
        s = json.load(open(summary_path, encoding='utf-8'))
        t = s['test']
        severity_rows.append(s.get('severity_summary') or severity_row(t, sev, args.seed))

        metrics_path = summary_path.parent / 'test_metrics.csv'
        query_counts = {}
        if metrics_path.exists():
            m = pd.read_csv(metrics_path, usecols=lambda c: c == 'family')
            if 'family' in m.columns:
                query_counts = {str(k): int(v) for k, v in m['family'].astype(str).value_counts().to_dict().items()}
        rows = s.get('scenario_summary') or scenario_rows(t, families, sev, args.seed, query_counts)
        scenario_all.extend(rows)
        source_dirs.append(str(summary_path.parent))

    for sev in args.severities:
        got = [str(r['scenario']) for r in scenario_all if int(r['severity']) == int(sev)]
        if set(got) != set(families) or len(got) != len(families):
            raise RuntimeError(f's{sev} scenario coverage mismatch expected={families} got={got}')

    stamp = args.run_stamp or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = base / 'summary' / f'{stamp}_seed{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    sev_csv = out / 'severity_summary.csv'
    scen_csv = out / 'scenario_summary.csv'
    pd.DataFrame(severity_rows).to_csv(sev_csv, index=False)
    pd.DataFrame(scenario_all).to_csv(scen_csv, index=False)
    (out / 'sources.json').write_text(json.dumps(source_dirs, indent=2), encoding='utf-8')

    print('\n' + '=' * 118)
    print('FINAL SEVERITY SUMMARY')
    print('=' * 118)
    for line in severity_table_lines(severity_rows):
        print(line)

    print('\n' + '=' * 154)
    print('FINAL SCENARIO SUMMARY')
    print('Scenario = missing-sensing family; metrics are micro-aggregated within that scenario.')
    print('=' * 154)
    for line in scenario_table_lines(scenario_all):
        print(line)

    print(f'\nSEVERITY_SUMMARY_CSV={sev_csv}')
    print(f'SCENARIO_SUMMARY_CSV={scen_csv}')


if __name__ == '__main__':
    main()
