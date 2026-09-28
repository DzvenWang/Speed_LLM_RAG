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
for p in [ROOT, REPO_ROOT/'Protocol', REPO_ROOT/'TraffiDent_SupportBench']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
sys.dont_write_bytecode = True

from experiment_common.cache import cache_dir, load_protocol_stage_metadata
from experiment_common.utils import load_yaml
from supportbench.dataset import SupportBench


def main() -> None:
    ap = argparse.ArgumentParser(description='Audit materialized ExperimentCommon support masks.')
    ap.add_argument('--config', default=str(ROOT/'configs'/'experiment_contract.yaml'))
    ap.add_argument('--stage', required=True, choices=['development','formal','evaluation'])
    ap.add_argument('--json-out', default=None)
    a = ap.parse_args()

    cfg = load_yaml(a.config)
    pm = load_protocol_stage_metadata(cfg, a.stage)
    out = cache_dir(cfg, a.stage, protocol_stage_hash=str(pm['stage_hash']))
    split = 'test' if a.stage == 'evaluation' else 'val'
    meta_path = out/f'{split}_manifest.csv'
    support_path = out/f'{split}_support_mask.npy'
    if not meta_path.exists() or not support_path.exists():
        raise FileNotFoundError(f'Missing cache inputs under {out}')
    meta = pd.read_csv(meta_path)
    support = np.load(support_path, mmap_mode='r').astype(bool)
    if len(meta) != len(support):
        raise RuntimeError(f'row mismatch manifest={len(meta)} support_mask={len(support)}')

    bench = SupportBench(cfg['experiment']['subgraph'], root=cfg['project']['benchmark_root'])
    rows = []
    families = list(cfg['experiment']['families'])
    for family in families:
        idx = np.flatnonzero(meta.family.astype(str).to_numpy() == str(family))
        if len(idx) == 0:
            continue
        fam_support = np.asarray(support[idx], bool)
        target_count = int(fam_support.sum())
        nonempty_queries = int(np.any(fam_support, axis=1).sum())
        structural_na = 0
        attainable_empty = 0
        max_src = []
        threshold = None
        if target_count == 0:
            for j in idx:
                cm = bench.predictive_context_metrics(bench.sample(str(meta.iloc[int(j)].view_id)))
                structural_na += int(cm['support_subset_status'] == 'N/A_STRUCTURAL')
                attainable_empty += int(cm['support_subset_status'] == 'EMPTY_ATTAINABLE')
                max_src.append(float(cm['max_source_missing_fraction']))
                threshold = float(cm['support_dependent_missing_threshold'])
            status = 'N/A_STRUCTURAL' if structural_na == len(idx) else 'EMPTY_ATTAINABLE'
        else:
            status = 'NONEMPTY'
            cm = bench.predictive_context_metrics(bench.sample(str(meta.iloc[int(idx[0])].view_id)))
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
        rows.append(row)
        print(
            f"CACHE_SUPPORT_SUBSET {str(family):<22} queries={len(idx):4d} "
            f"targets={target_count:6d} nonempty_q={nonempty_queries:4d} status={status}"
            + (f" max_src_missing={row['max_source_missing_fraction_max_checked']:.4f} threshold={row['support_threshold']:.4f}" if row['max_source_missing_fraction_max_checked'] is not None else '')
        )

    bad = [r for r in rows if r['status'] == 'EMPTY_ATTAINABLE']
    report = {
        'cache_dir': str(out),
        'stage': a.stage,
        'split': split,
        'protocol_stage_hash': str(pm['stage_hash']),
        'rows': rows,
        'status': 'FAIL' if bad else 'PASS',
    }
    if a.json_out:
        op = Path(a.json_out)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('JSON_OUT=' + str(op))
    if bad:
        raise SystemExit('CACHE SUPPORT SUBSET AUDIT FAILED: ' + ', '.join(r['family'] for r in bad))
    print('EXPERIMENTCOMMON SUPPORT SUBSET AUDIT PASS')


if __name__ == '__main__':
    main()
