#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backbone.utils import load_yaml
from backbone.evaluation_reporting import (
    scenario_rows,
    severity_row,
    aggregate_mean,
    severity_table_lines,
    scenario_table_lines,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'configs' / 'backbone.yaml'))
    a = ap.parse_args()
    cfg = load_yaml(a.config)
    exp = Path(cfg['project']['exp_root'])
    sg = cfg['experiment']['subgraph']
    seeds = [int(x) for x in cfg['experiment']['formal_seeds']]
    families = [str(x) for x in cfg['experiment']['families']]

    severity_runs = []
    scenario_runs = []
    for sev in [int(x) for x in cfg['experiment']['evaluation_severities']]:
        root = exp / 'PatchSTG' / sg / 'evaluation' / f's{sev}'
        for seed in seeds:
            fs = [] if not root.exists() else list(root.glob(f'*_seed{seed}/test_summary.json'))
            if not fs:
                continue
            f = max(fs, key=lambda p: p.stat().st_mtime)
            s = json.load(open(f, encoding='utf-8'))
            t = s['test']
            sev_row = s.get('severity_summary') or severity_row(t, sev, seed)
            sev_row = dict(sev_row)
            sev_row['run_dir'] = str(f.parent)
            severity_runs.append(sev_row)

            if s.get('scenario_summary'):
                rows = [dict(x) for x in s['scenario_summary']]
            else:
                rows = scenario_rows(t, families, sev, seed)
            for row in rows:
                row['run_dir'] = str(f.parent)
            scenario_runs.extend(rows)

    if not severity_runs:
        raise SystemExit('No evaluation summaries found')

    sev_df = pd.DataFrame(severity_runs).sort_values(['severity', 'seed'])
    scen_df = pd.DataFrame(scenario_runs).sort_values(['severity', 'scenario', 'seed'])
    sev_df.to_csv(exp / 'backbone_evaluation_runs.csv', index=False)
    scen_df.to_csv(exp / 'backbone_evaluation_scenario_runs.csv', index=False)

    sev_agg = aggregate_mean(severity_runs, ['severity'])
    scen_agg = aggregate_mean(scenario_runs, ['severity', 'scenario'])
    pd.DataFrame(sev_agg).sort_values('severity').to_csv(exp / 'backbone_evaluation_summary.csv', index=False)
    pd.DataFrame(scen_agg).sort_values(['severity', 'scenario']).to_csv(exp / 'backbone_evaluation_scenario_summary.csv', index=False)

    print('\n' + '=' * 118)
    print('FINAL SEVERITY SUMMARY (mean across available formal seeds)')
    print('=' * 118)
    for line in severity_table_lines(sev_agg):
        print(line)

    print('\n' + '=' * 154)
    print('FINAL SCENARIO SUMMARY (mean across available formal seeds)')
    print('Scenario = missing-sensing family; metrics are micro-aggregated within each scenario before seed averaging.')
    print('=' * 154)
    for line in scenario_table_lines(scen_agg):
        print(line)

    print(f'\nSEVERITY_RUNS_CSV={exp / "backbone_evaluation_runs.csv"}')
    print(f'SEVERITY_SUMMARY_CSV={exp / "backbone_evaluation_summary.csv"}')
    print(f'SCENARIO_RUNS_CSV={exp / "backbone_evaluation_scenario_runs.csv"}')
    print(f'SCENARIO_SUMMARY_CSV={exp / "backbone_evaluation_scenario_summary.csv"}')


if __name__ == '__main__':
    main()
