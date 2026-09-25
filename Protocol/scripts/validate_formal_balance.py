#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared_protocol.utils import load_yaml, read_json


def main() -> None:
    ap = argparse.ArgumentParser(description='Validate exact-balanced s40 Formal training protocol.')
    ap.add_argument('--config', default=str(ROOT / 'configs' / 'protocol_s40.yaml'))
    a = ap.parse_args()
    cfg = load_yaml(a.config)
    dataset = cfg['experiment']['dataset']
    name = cfg['protocol']['name']
    sd = Path(cfg['project']['artifact_root']) / dataset / name / 'formal'
    frozen = sd / 'FROZEN.json'
    train_csv = sd / 'train_views.csv'
    if not frozen.exists() or not train_csv.exists():
        raise SystemExit(f'Formal protocol is not frozen: {sd}')

    df = pd.read_csv(train_csv)
    fams = list(cfg['experiment']['families'])
    total = int(cfg['stages']['formal']['train_size'])
    if total % len(fams) != 0:
        raise AssertionError(f'Formal train_size={total} is not divisible by {len(fams)} families.')
    per_family = total // len(fams)
    counts = {f: int((df.family.astype(str) == f).sum()) for f in fams}
    result = {
        'protocol_name': name,
        'stage': 'formal',
        'stage_hash': read_json(frozen)['stage_hash'],
        'rows': int(len(df)),
        'unique_groups': int(df.view_group_id.astype(str).nunique()),
        'expected_per_family': int(per_family),
        'family_counts': counts,
    }
    print(json.dumps(result, indent=2))
    if len(df) != total:
        raise AssertionError(f'Expected {total} Formal train rows, got {len(df)}.')
    if result['unique_groups'] != total:
        raise AssertionError(f"Expected {total} unique groups, got {result['unique_groups']}.")
    bad = {f: c for f, c in counts.items() if c != per_family}
    if bad:
        raise AssertionError(f'Formal family balance failed: expected {per_family} each, got {bad}')
    if not (df.severity.astype(float).round(8) == float(cfg['experiment']['severity'])).all():
        raise AssertionError('Formal train contains wrong severity.')
    print('FORMAL BALANCE VALIDATION PASS')


if __name__ == '__main__':
    main()
