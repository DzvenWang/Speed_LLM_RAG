#!/usr/bin/env python3
"""Readable fixed-checkpoint Retrieval interventions for Validation and Test.

Validation uses the pilot/formal Validation cache (training mask severity, normally s40).
Test is evaluated separately at each configured SupportBench severity.
These interventions neither retrain the model nor select a checkpoint.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

MODES=('full','similarity_memory','utility_off','random_memory','no_memory')


def _gain(full, comparator):
    baseline=float(comparator['MAE'])
    if baseline <= 0: return 0.0
    return 100.0*(baseline-float(full['MAE']))/baseline


def build_report(rows):
    if not rows:
        raise ValueError('Empty Retrieval Attribution results')
    groups=defaultdict(dict)
    for row in rows:
        split=row['split']
        severity=row.get('severity','')
        severity='' if severity is None else str(severity)
        key=(split,severity)
        mode=row['mode']
        if mode in groups[key]:
            raise ValueError(f'Duplicate mode for {key}: {mode}')
        groups[key][mode]=row
    for key, modes in groups.items():
        if 'full' not in modes:
            raise ValueError(f'Missing full Retrieval result for {key}')
    lines=['# FARE — Retrieval Attribution: Validation and Test','',
           'All interventions share ONE checkpoint chosen on Validation. '
           'Positive gain means Full/Real Retrieval has lower MAE than the comparator.','',
           '**Test is exploratory in Pilot. Do not tune, select checkpoints, or choose a model version using these Test results.**','',
           'The Validation split is from the training-protocol cache (normally s40). '
           'The Test split uses separate evaluation caches at s20, s40, s60 and s90.','',
           '| Split | Severity | Mode | MAE | RMSE | Full advantage (MAE) |',
           '|---|---|---|---:|---:|---:|']
    for (split,severity), modes in sorted(groups.items(),key=lambda g:(0 if g[0][0]=='val' else 1,int(g[0][1]) if g[0][1] else -1)):
        base=modes['full']
        for mode in (x for x in MODES if x in modes):
            value=modes[mode]
            gain=0.0 if mode=='full' else _gain(base,value)
            tag='Training-mask validation' if split=='val' else f's{int(severity):02d}'
            lines.append(f'| {split} | {tag} | {mode} | {float(value["MAE"]):.5f} | '
                         f'{float(value["RMSE"]):.5f} | {gain:+.2f}% |')
    tests=sorted((k for k in groups if k[0]=='test'),key=lambda k:int(k[1]))
    if tests:
        lines += ['', '## Test severity macro-average (unweighted average across evaluated severities)', '',
                  '| Mode | Avg MAE | Avg RMSE | Full advantage (Avg MAE) |',
                  '|---|---:|---:|---:|']
        modes=set.intersection(*(set(groups[k]) for k in tests))
        avgs={mode:{metric:sum(float(groups[k][mode][metric]) for k in tests)/len(tests)
                    for metric in ('MAE','RMSE')} for mode in modes}
        for mode in (x for x in MODES if x in avgs):
            gain=0.0 if mode=='full' else _gain(avgs['full'],avgs[mode])
            lines.append(f'| {mode} | {avgs[mode]["MAE"]:.5f} | {avgs[mode]["RMSE"]:.5f} | {gain:+.2f}% |')
        lines += ['', 'This macro-average is NOT a pooled-observation MAE or RMSE. '
                  'It weights every evaluated Test severity equally.']
    lines += ['', '## Interpretation', '',
              'The Fixed-Checkpoint retrieval interventions can indicate whether query-matched evidence adds value '
              'to this checkpoint. They do not isolate gains from extra parameters or training capacity. '
              'Use matched retrained ablations and multiple seeds for paper conclusions.',
              '', 'Forecast Utility teachers are region-level historical-future proxies, '
              'not oracle node-wise forecasting utility.']
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


if __name__=='__main__':main()
