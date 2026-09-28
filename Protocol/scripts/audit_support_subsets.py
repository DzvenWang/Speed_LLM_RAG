#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
for p in [ROOT, REPO_ROOT/'TraffiDent_SupportBench']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
sys.dont_write_bytecode = True

from shared_protocol.consumer import load_frozen_stage
from shared_protocol.utils import load_yaml
from supportbench.dataset import SupportBench


def main() -> None:
    ap = argparse.ArgumentParser(description='Audit frozen Protocol support subsets without mutating frozen artifacts.')
    ap.add_argument('--config', default=str(ROOT/'configs'/'protocol_s20.yaml'))
    ap.add_argument('--stage', default='evaluation', choices=['development','formal','evaluation'])
    ap.add_argument('--json-out', default=None)
    a = ap.parse_args()

    cfg = load_yaml(a.config)
    cfg['_config_path'] = str(Path(a.config).resolve())
    manifest, views = load_frozen_stage(
        cfg['project']['artifact_root'], cfg['experiment']['dataset'],
        cfg['protocol']['name'], a.stage, require_evaluation_access=False,
    )
    bench = SupportBench(cfg['experiment']['dataset'], root=cfg['project']['benchmark_root'])
    stage_dir = Path(cfg['project']['artifact_root'])/cfg['experiment']['dataset']/cfg['protocol']['name']/a.stage
    split = 'test' if a.stage == 'evaluation' else 'val'
    mask_path = stage_dir/f'{split}_evaluation_masks.npz'
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)
    z = np.load(mask_path, allow_pickle=False)
    support = np.asarray(z['support_mask'], bool)
    df = views[split].reset_index(drop=True)
    if len(df) != len(support):
        raise RuntimeError(f'row mismatch views={len(df)} support_mask={len(support)}')

    report = []
    for family in list(cfg['experiment']['families']):
        idx = np.flatnonzero(df.family.astype(str).to_numpy() == str(family))
        if len(idx) == 0:
            continue
        fam_support = support[idx]
        target_count = int(fam_support.sum())
        nonempty_queries = int(np.any(fam_support, axis=1).sum())
        structural_na = 0
        attainable_empty = 0
        max_src = []
        threshold = None
        if target_count == 0:
            for j in idx:
                cm = bench.predictive_context_metrics(bench.sample(str(df.iloc[int(j)].view_id)))
                structural_na += int(cm['support_subset_status'] == 'N/A_STRUCTURAL')
                attainable_empty += int(cm['support_subset_status'] == 'EMPTY_ATTAINABLE')
                max_src.append(float(cm['max_source_missing_fraction']))
                threshold = float(cm['support_dependent_missing_threshold'])
            status = 'N/A_STRUCTURAL' if structural_na == len(idx) else 'EMPTY_ATTAINABLE'
        else:
            status = 'NONEMPTY'
            # One representative view is enough for reporting the configured threshold.
            cm = bench.predictive_context_metrics(bench.sample(str(df.iloc[int(idx[0])].view_id)))
            threshold = float(cm['support_dependent_missing_threshold'])
            max_src.append(float(cm['max_source_missing_fraction']))

        row = {
            'severity': float(cfg['experiment']['severity']),
            'family': str(family),
            'queries': int(len(idx)),
            'support_target_count': target_count,
            'support_nonempty_queries': nonempty_queries,
            'structural_na_queries': int(structural_na),
            'empty_attainable_queries': int(attainable_empty),
            'max_source_missing_fraction_max_checked': float(max(max_src)) if max_src else None,
            'support_threshold': float(threshold) if threshold is not None else None,
            'status': status,
        }
        report.append(row)
        print(
            f"SUPPORT_SUBSET {str(family):<22} queries={len(idx):4d} "
            f"targets={target_count:6d} nonempty_q={nonempty_queries:4d} "
            f"status={status}"
            + (f" max_src_missing={row['max_source_missing_fraction_max_checked']:.4f} threshold={row['support_threshold']:.4f}" if row['max_source_missing_fraction_max_checked'] is not None else '')
        )

    bad = [r for r in report if r['status'] == 'EMPTY_ATTAINABLE']
    out = {
        'protocol_name': cfg['protocol']['name'],
        'stage': a.stage,
        'stage_hash': manifest['stage_hash'],
        'split': split,
        'rows': report,
        'status': 'FAIL' if bad else 'PASS',
    }
    if a.json_out:
        op = Path(a.json_out)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(out, indent=2), encoding='utf-8')
        print('JSON_OUT=' + str(op))
    if bad:
        raise SystemExit('SUPPORT SUBSET AUDIT FAILED: ' + ', '.join(r['family'] for r in bad))
    print('PROTOCOL SUPPORT SUBSET AUDIT PASS')


if __name__ == '__main__':
    main()
