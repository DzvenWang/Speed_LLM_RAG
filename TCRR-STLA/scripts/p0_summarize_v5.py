#!/usr/bin/env python3
"""Small cross-target P0 readout and block-bootstrap descriptive uncertainty.

Does NOT update weights, calibration, thresholds or model selection. Test-label
statistics are research reporting only, never admissible for tuning v5.4.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

FIELDS=(
    ('c1_gain','C1 M000-M100 MAE gain'),
    ('c2_gain','C2 M100-M110 MAE gain'),
    ('c3_gain','C3 M110-M111 MAE gain'),
    ('c3_harm_rate','C3 harm position rate'),
    ('oracle_single_gain','Single candidate hindsight oracle gain over Backbone'),
    ('oracle_single_minus_m110','Signed single-oracle minus M110 mixture gap'),
    ('harmful_support_weight_share','Candidate weight on harmful support'),
)


def load_csv(path):
    with Path(path).open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def weighted_mean(rows,key):
    if not rows:return None
    w=np.asarray([float(r['valid_positions']) for r in rows])
    x=np.asarray([float(r[key]) for r in rows])
    valid=(w>0)&np.isfinite(x)
    return float(np.average(x[valid],weights=w[valid])) if valid.any() else None


def block_bootstrap(rows,key,seed=10,block_size=24,repeats=400):
    """Resamples contiguous dataset-index blocks, not individual forecast pixels."""
    if not rows:return None,None
    rows=sorted(rows,key=lambda r:int(r['dataset_index']))
    groups={}
    for r in rows:
        groups.setdefault(int(r['dataset_index'])//block_size,[]).append(r)
    groups=list(groups.values())
    if len(groups)<2:return None,None
    vals=np.asarray([[sum(float(r[key])*float(r['valid_positions']) for r in g),
                      sum(float(r['valid_positions']) for r in g)] for g in groups],dtype=np.float64)
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(groups),size=(repeats,len(groups)))
    sampled=vals[draws].sum(axis=1)
    res=sampled[:,0]/np.clip(sampled[:,1],1,None)
    return tuple(float(x) for x in np.percentile(res,[2.5,97.5]))


def locate(run,section):
    root=Path(run)/'evaluation'
    for label,path in (
        (f'validation_{section}',root/'validation'/'automated'/f'p0_deep_{section}'/'correct__correct'/'validation'/'p0_deep'),
        *((f'test_s{s}',root/'test'/'automated'/'p0_deep_test_readonly'/'correct__correct'/f's{s}'/'p0_deep') for s in (20,40,60,90)),
    ):
        if (path/'summary.json').is_file() and (path/'per_query.csv').is_file():
            yield label,path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='.',help='Speed_LLM_RAG root')
    ap.add_argument('--run-stamp',default='20261028_155629_seed10_full')
    ap.add_argument('--validation-section',default='selection',choices=('selection','calibration','check'))
    ap.add_argument('--bootstrap-block-size',type=int,default=24)
    ap.add_argument('--bootstrap-repeats',type=int,default=400)
    ap.add_argument('--output',default=None)
    a=ap.parse_args()
    if a.bootstrap_block_size<1 or a.bootstrap_repeats<50:ap.error('Invalid bootstrap configuration')
    root=Path(a.root).expanduser().resolve()
    out=Path(a.output).resolve() if a.output else root/'TCRR-STLA_exps'/'p0_diagnostics_summary'
    out.mkdir(parents=True,exist_ok=True)
    report=[]
    prior_negative=[]
    prior_variants=[]
    for target in ('speed','flow'):
        run=(root/'TCRR-STLA_exps'/target/'incomplete'/'TraffiDent_Net_06'/'v5'/'formal'/a.run_stamp)
        old=run/'evaluation'/'validation'/'automated'
        for name,collector in (('negative_control_comparison.csv',prior_negative),
                               ('c3_variant_comparison.csv',prior_variants)):
            f=old/name
            if f.is_file():
                collector.extend(dict(target=target,
                                      source_partition='existing_chronological_check_512query_diagnostic',
                                      **row) for row in load_csv(f))
        for group,path in locate(run,a.validation_section):
            summary=json.loads((path/'summary.json').read_text())
            rows=load_csv(path/'per_query.csv')
            if summary['queries']!=len(rows):
                raise RuntimeError(f'Query count mismatch: {path}')
            for key,description in FIELDS:
                mean=weighted_mean(rows,key)
                ci=block_bootstrap(rows,key,block_size=a.bootstrap_block_size,
                                   repeats=a.bootstrap_repeats)
                report.append(dict(target=target,partition=group,metric=key,
                    interpretation=description,estimate=mean,
                    bootstrap_95_low=ci[0],bootstrap_95_high=ci[1],
                    queries=len(rows),valid_positions=summary['valid_positions'],
                    source=str(path)))
    if not report:
        raise FileNotFoundError('No P0 deep outputs found; run the PBS diagnostics first')
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(report[0]))
        writer.writeheader();writer.writerows(report)
    for name,collector in (('existing_negative_controls.csv',prior_negative),
                           ('existing_c3_variants.csv',prior_variants)):
        if collector:
            with (out/name).open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(collector[0]))
                writer.writeheader();writer.writerows(collector)
    text=['# TCRR v5.3.0 P0 paired diagnosis','',
          'All gains are positive for lower MAE. Original Speed/Flow target units.',
          'Single candidate oracle is a hindsight ceiling against Backbone, NOT a deployable predictor.',
          'The signed oracle-minus-mixture gap may be negative.',
          'Intervals use contiguous query-index block bootstrap; overlapping windows or other correlations may still affect uncertainty.',
          'Test partition findings are post-hoc description only. Never tune C2/C3 on test labels.',
          'Existing negative controls and C3 variants are check-subset experiments and must not be compared directly to full TEST MAE.',
          f'Existing negatives: {len(prior_negative)} rows; C3 variants: {len(prior_variants)} rows.','',
          f'Bootstrap: {a.bootstrap_repeats} repetitions, block size {a.bootstrap_block_size} queries.','',
          '|Target|Partition|Metric|Estimate|95% block-bootstrap interval|','|---|---|---|---:|---:|']
    for r in report:
        ci=('n/a' if r['bootstrap_95_low'] is None else
            f"[{r['bootstrap_95_low']:.5f}, {r['bootstrap_95_high']:.5f}]")
        text.append(f"|{r['target']}|{r['partition']}|{r['metric']}|{r['estimate']:.6f}|{ci}|")
    (out/'P0_readout.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(f'V53_P0_SUMMARIZE_PASS rows={len(report)} output={out}',flush=True)

if __name__=='__main__':main()
