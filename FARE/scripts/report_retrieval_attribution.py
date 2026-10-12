#!/usr/bin/env python3
"""Readable fixed-checkpoint Retrieval interventions, canonical MAPE protocol."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

MODES=('full','similarity_memory','utility_off','random_memory',
       'cross_query_shuffle','weight_shuffle','null_evidence','retrieval_off','no_memory')


def _fmt(value, decimals=2):
    return 'N/A' if value is None or not math.isfinite(float(value)) else f'{float(value):.{decimals}f}'


def _macro(rows, metric):
    values=[r.get(metric) for r in rows]
    if not values or any(v is None or not math.isfinite(float(v)) for v in values):
        return None
    return sum(float(v) for v in values)/len(values)


def _gain(full, comparator):
    b=comparator.get('MAE')
    f=full.get('MAE')
    if b is None or f is None or not math.isfinite(float(b)) or float(b)<=0 or not math.isfinite(float(f)):
        return float('nan')
    return 100.0*(float(b)-float(f))/float(b)


def build_report(rows):
    if not rows:
        raise ValueError('Empty Retrieval Attribution results')
    groups=defaultdict(dict)
    for row in rows:
        split=row['split']
        severity='' if row.get('severity') is None else str(row.get('severity',''))
        key=(split,severity)
        mode=row['mode']
        if mode in groups[key]:
            raise ValueError(f'Duplicate mode for {key}: {mode}')
        if row.get('mape_protocol') != 'raw_target_percent_exclude_abs_le_1_v1':
            raise ValueError(f'Ablation metrics were not computed with the shared Base_SupportBench protocol: {key}/{mode}')
        groups[key][mode]=row
    for key,modes in groups.items():
        if 'full' not in modes:
            raise ValueError(f'Missing full Retrieval result for {key}')
    lines=[
        '# FARE Retrieval Attribution: Validation and Test','',
        'All interventions share the same Validation-selected checkpoint. '
        'A positive gain means Real Retrieval has lower MAE than that comparator.','',
        '**Test is exploratory: do not select checkpoints using Test results.**','',
        'Raw-unit MAPE uses the shared Base_SupportBench protocol: |truth| > 1.0; '
        'MAPE is cell-level micro-aggregated within each severity.','',
        '| Split | Severity | Mode | MAE | RMSE | MAPE (%) | MAPE count | MAPE coverage | Real advantage (MAE) |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for (split,severity),modes in sorted(groups.items(),key=lambda g:(0 if g[0][0]=='val' else 1,int(g[0][1]) if g[0][1] else -1)):
        real=modes['full']
        for mode in (x for x in MODES if x in modes):
            value=modes[mode]
            gain=0.0 if mode=='full' else _gain(real,value)
            tag='Training-mask validation' if split=='val' else f's{int(severity):02d}'
            lines.append(
                f'| {split} | {tag} | {mode} | {_fmt(value["MAE"],5)} | '
                f'{_fmt(value["RMSE"],5)} | {_fmt(value.get("MAPE"))} | '
                f'{value.get("MAPE_count","N/A")} | {_fmt(value.get("MAPE_coverage"),3)} | '
                f'{gain:+.2f}% |'
            )
    tests=sorted((k for k in groups if k[0]=='test'),key=lambda k:int(k[1]))
    if tests:
        lines.extend(['','## Test severity macro-average (equal weight across severities)','',
                      '| Mode | Avg MAE | Avg RMSE | Avg MAPE (%) | Real advantage (Avg MAE) |',
                      '|---|---:|---:|---:|---:|'])
        modes=set.intersection(*(set(groups[k]) for k in tests))
        avgs={mode:{metric:_macro([groups[k][mode] for k in tests],metric)
                    for metric in ('MAE','RMSE','MAPE')}
              for mode in modes}
        for mode in (x for x in MODES if x in avgs):
            gain=0.0 if mode=='full' else _gain(avgs['full'],avgs[mode])
            lines.append(
                f'| {mode} | {_fmt(avgs[mode]["MAE"],5)} | '
                f'{_fmt(avgs[mode]["RMSE"],5)} | '
                f'{_fmt(avgs[mode]["MAPE"])} | {gain:+.2f}% |'
            )
        lines.extend(['','These are unweighted across-severity averages, not pooled Test metrics.'])
    lines.extend(['','## Interpretation','',
                  'Fixed-checkpoint interventions measure the sensitivity of this checkpoint to evidence. '
                  'A matched retrained No-memory model and multiple seeds are needed for a stronger attribution claim.',
                  '', 'The Utility Teacher uses a purged later TRAIN time-block; it is not full K-fold OOF.'])
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    rows=json.loads(Path(args.input).read_text(encoding='utf-8'))
    report=build_report(rows)
    dest=Path(args.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(report,encoding='utf-8')
    print('FARE_RETRIEVAL_REPORT '+json.dumps({'report':str(dest),
          'splits':sorted({r['split'] for r in rows}), 'rows':len(rows)}),flush=True)

if __name__=='__main__':
    main()
