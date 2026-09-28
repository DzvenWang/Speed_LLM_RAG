#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from supportbench.config import load_config
from supportbench.dataset import SupportBench


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Audit support-dependent subset applicability by severity/family.'
    )
    ap.add_argument('--config', default=str(ROOT/'configs'/'supportbench.yaml'))
    ap.add_argument('--subgraph', default='TraffiDent_Net_06')
    ap.add_argument('--split', default='test', choices=['train','val','test'])
    ap.add_argument('--max-views-per-cell', type=int, default=256)
    ap.add_argument('--json-out', default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    bench = SupportBench(a.subgraph, root=cfg['project']['benchmark_root'])
    views = bench.views_for(a.split).copy()
    families = list(cfg['sensing']['general_families'])
    severities = [float(x) for x in cfg['sensing']['severities']]
    rows = []

    for severity in severities:
        for family in families:
            df = views[
                views.family.astype(str).eq(family)
                & np.isclose(views.severity.astype(float), severity)
                & views.distribution.astype(str).eq('id')
            ].head(max(1, int(a.max_views_per_cell)))
            if len(df) == 0:
                rows.append({
                    'severity': severity, 'family': family, 'checked_views': 0,
                    'support_target_count': 0, 'support_nonempty_queries': 0,
                    'structural_na_queries': 0, 'empty_attainable_queries': 0,
                    'max_source_missing_fraction_max': None,
                    'support_threshold': float(cfg['dependency']['support_dependent_missing_threshold']),
                    'status': 'NO_VIEWS',
                })
                continue

            support_target_count = 0
            support_nonempty_queries = 0
            structural_na_queries = 0
            empty_attainable_queries = 0
            max_source_missing = []
            threshold = None
            for vid in df.view_id.astype(str):
                cm = bench.predictive_context_metrics(bench.sample(vid))
                c = int(cm['support_dependent_target_count'])
                support_target_count += c
                support_nonempty_queries += int(c > 0)
                structural_na_queries += int(cm['support_subset_status'] == 'N/A_STRUCTURAL')
                empty_attainable_queries += int(cm['support_subset_status'] == 'EMPTY_ATTAINABLE')
                max_source_missing.append(float(cm['max_source_missing_fraction']))
                threshold = float(cm['support_dependent_missing_threshold'])

            if support_target_count > 0:
                status = 'NONEMPTY'
            elif structural_na_queries == len(df):
                status = 'N/A_STRUCTURAL'
            else:
                status = 'EMPTY_ATTAINABLE'

            rows.append({
                'severity': severity,
                'family': family,
                'checked_views': int(len(df)),
                'support_target_count': int(support_target_count),
                'support_nonempty_queries': int(support_nonempty_queries),
                'structural_na_queries': int(structural_na_queries),
                'empty_attainable_queries': int(empty_attainable_queries),
                'max_source_missing_fraction_min': float(np.min(max_source_missing)),
                'max_source_missing_fraction_mean': float(np.mean(max_source_missing)),
                'max_source_missing_fraction_max': float(np.max(max_source_missing)),
                'support_threshold': float(threshold),
                'status': status,
            })

    print('SUPPORT SUBSET AUDIT')
    print('severity family                 checked targets nonempty_q structural_na empty_attainable max_src_missing threshold status')
    for r in rows:
        mx = '-' if r.get('max_source_missing_fraction_max') is None else f"{r['max_source_missing_fraction_max']:.4f}"
        print(
            f"s{int(round(r['severity']*100)):02d}     {r['family']:<22} "
            f"{r['checked_views']:>7} {r['support_target_count']:>7} "
            f"{r['support_nonempty_queries']:>10} {r['structural_na_queries']:>13} "
            f"{r['empty_attainable_queries']:>16} {mx:>15} "
            f"{r['support_threshold']:.4f} {r['status']}"
        )

    out = {'subgraph': a.subgraph, 'split': a.split, 'rows': rows}
    if a.json_out:
        jp = Path(a.json_out)
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(out, indent=2), encoding='utf-8')
        print('JSON_OUT=' + str(jp))

    bad = [r for r in rows if r['status'] == 'EMPTY_ATTAINABLE']
    if bad:
        raise SystemExit(
            'SUPPORT SUBSET AUDIT FAILED: empty support subset despite an attainable threshold: '
            + ', '.join(f"s{int(round(r['severity']*100)):02d}/{r['family']}" for r in bad)
        )
    print('SUPPORT SUBSET AUDIT PASS')


if __name__ == '__main__':
    main()
