#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from backbone.utils import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'configs' / 'backbone_selection.yaml'))
    ap.add_argument('--severities', nargs='+', default=['20','40','60','90'])
    a = ap.parse_args()
    cfg = load_yaml(a.config)
    exp = Path(cfg['project']['exp_root'])
    rows = []
    for sev in a.severities:
        p = exp / f'frozen_test_s{sev}' / 'test_summary.json'
        if not p.exists():
            print('SKIP missing', p)
            continue
        obj = json.loads(p.read_text())
        t = obj['test']
        rows.append({
            'severity': float(obj.get('configured_severity', int(sev)/100.0)),
            'tag': f's{sev}',
            'model': obj.get('selected_model'),
            'MAE': t.get('MAE'),
            'RMSE': t.get('RMSE'),
            'directly_masked_MAE': t.get('directly_masked_MAE'),
            'support_dependent_MAE': t.get('support_dependent_MAE'),
            'H12_MAE': t.get('H12_MAE'),
            'H12_support_dependent_MAE': t.get('H12_support_dependent_MAE'),
            'protocol': obj.get('evaluation_protocol'),
        })
    if not rows:
        raise SystemExit('No severity evaluation results found.')
    df = pd.DataFrame(rows).sort_values('severity')
    out = exp / 'severity_robustness_summary.csv'
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print('WROTE', out)


if __name__ == '__main__':
    main()
