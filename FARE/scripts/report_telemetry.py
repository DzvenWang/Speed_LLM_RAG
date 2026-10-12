#!/usr/bin/env python3
"""Combine independent PBS GPU/CPU samples with FARE epoch diagnostics.

Reports are read-only. They never load data, run FARE inference, or alter
model checkpoints. Incomplete JSONL is tolerated (for interrupted PBS jobs).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from pathlib import Path

from resource_monitor import read_jsonl, summarize


def _parse_utc(value):
    try:return dt.datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()
    except (AttributeError,ValueError):return None


def _mean(values):
    valid=[float(v) for v in values if v is not None]
    return round(sum(valid)/len(valid),3) if valid else None


def _max(values):
    valid=[float(v) for v in values if v is not None]
    return round(max(valid),3) if valid else None


def resolve_run_dir(live_log):
    path=Path(live_log)
    if not path.is_file():return None
    lines=path.read_text(encoding='utf-8',errors='replace').splitlines()
    # Scan from end; FARE_TRAIN_PASS is emitted on successful completion.
    for text in reversed(lines):
        match=re.search(r'FARE_TRAIN_PASS run_dir=(.*?) candidate_checkpoint=',text)
        if match:return Path(match.group(1).strip())
        match=re.search(r'FARE_DIAGNOSTICS_CONFIG (\{.*\})',text)
        if match:
            try:return Path(json.loads(match.group(1))['run_dir'])
            except (KeyError,ValueError):pass
    return None


def report(run_dir,resource_prefix=None):
    run_dir=Path(run_dir)
    diag=run_dir/'diagnostics';diag.mkdir(parents=True,exist_ok=True)
    epochs=read_jsonl(diag/'epoch_diagnostics.jsonl')
    sample_path=Path(f'{resource_prefix}.jsonl') if resource_prefix else None
    resources=read_jsonl(sample_path) if sample_path else []
    aggregate=summarize(resources)
    per_epoch=[]
    for epoch in epochs:
        res=epoch.get('resource',{})
        start=_parse_utc(res.get('utc_start'))
        end=_parse_utc(res.get('utc_end'))
        matching=[sample for sample in resources
                  if start is not None and end is not None and
                  start<=sample.get('unix_time',-1)<=end]
        row={'phase':epoch['phase'],'epoch':epoch['epoch'],
             'resource_samples':len(matching),
             'gpu_matched_samples':sum(bool(x.get('gpu_matched')) for x in matching),
             'gpu_util_mean_pct':_mean(x.get('gpu_util_pct') for x in matching),
             'gpu_memory_peak_mb':_max(x.get('gpu_memory_used_mb') for x in matching),
             'cpu_cores_mean':_mean(x.get('cpu_cores') for x in matching),
             'process_rss_peak_gb':_max(x.get('process_rss_gb') for x in matching),
             'train_samples_per_second':res.get('train_samples_per_second'),
             'train_peak_cuda_allocated_gb':res.get('train_peak_cuda_allocated_gb'),
             'val_peak_cuda_allocated_gb':res.get('val_peak_cuda_allocated_gb'),
             'val_candidate_MAE_gain':epoch.get('val',{}).get('candidate_MAE_gain'),
             'val_final_MAE_gain':epoch.get('val',{}).get('final_MAE_gain'),
             'val_final_harmful_fraction':epoch.get('val',{}).get('final_harmful_fraction'),
             'val_retrieval_effective_topk':epoch.get('val',{}).get('retrieval_effective_topk'),
             'val_gate_near_zero_fraction':epoch.get('val',{}).get('gate_near_zero_fraction')}
        per_epoch.append(row)
    if per_epoch:
        with (diag/'epoch_resource.csv').open('w',encoding='utf-8',newline='') as fh:
            writer=csv.DictWriter(fh,fieldnames=list(per_epoch[0]))
            writer.writeheader();writer.writerows(per_epoch)
    stage_summary=[]
    stage_path=run_dir/'phase_summary.json'
    if stage_path.exists():
        try:stage_summary=json.loads(stage_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:pass
    warnings=[]
    util=aggregate.get('gpu_util_pct',{}).get('mean')
    if util is not None and util<45:
        warnings.append('Average matched-GPU utilization is below 45%; profile data loading, CPU work, and small kernels before changing the model.')
    late=[x for x in per_epoch if x['phase'] in ('integration','control')]
    if late:
        harmful=[x['val_final_harmful_fraction'] for x in late if x['val_final_harmful_fraction'] is not None]
        if harmful and _mean(harmful)>.4:
            warnings.append('Validation correction harm fraction is high; inspect gate calibration and candidate relevance.')
    report_json={'run_dir':str(run_dir),'resource_prefix':str(resource_prefix) if resource_prefix else None,
                 'epoch_count':len(epochs),'resource':aggregate,'per_epoch':per_epoch,
                 'phase_summary':stage_summary,'interpretation_flags_not_conclusions':warnings,
                 'caveat':'Epoch diagnosis is sampled on current train/val split (typically s40), NOT separate s20/s60/s90 test evaluation.'}
    (diag/'telemetry_report.json').write_text(json.dumps(report_json,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    lines=['# FARE diagnostic and resource report','',
           f'- Run: `{run_dir}`',f'- Completed/recorded epochs: {len(epochs)}',
           f'- Hardware resource snapshots: {aggregate.get("sample_count",0)}',
           f'- Matched-GPU snapshots: {aggregate.get("matched_gpu_samples",0)}','',
           '## Overall resource statistics','',
           '| Metric | Mean | P95 | Max |','|---|---:|---:|---:|']
    for key,label in [('gpu_util_pct','GPU utilization (%)'),('gpu_memory_used_mb','GPU memory used (MiB)'),
                      ('cpu_cores','Job CPU cores'),('process_rss_gb','Job process RSS (GB)'),('gpu_power_w','GPU power (W)')]:
        values=aggregate.get(key,{})
        lines.append(f'| {label} | {values.get("mean")} | {values.get("p95")} | {values.get("max")} |')
    lines+=['','## Per-epoch trends','',
            '| Phase | Epoch | Samples/sec | GPU avg % | GPU peak MiB | Val candidate MAE gain | Val final MAE gain |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for entry in per_epoch:
        lines.append(f'| {entry["phase"]} | {entry["epoch"]} | {entry["train_samples_per_second"]} | '
                     f'{entry["gpu_util_mean_pct"]} | {entry["gpu_memory_peak_mb"]} | '
                     f'{entry["val_candidate_MAE_gain"]} | {entry["val_final_MAE_gain"]} |')
    lines+=['','## Diagnostic flags (hypotheses only)','']
    lines.extend(f'- {message}' for message in warnings)
    if not warnings:lines.append('- No rule-based flag triggered, or insufficient resource/diagnostic samples.')
    lines+=['','These online validation diagnostics use the current train/val sensing setup; '
            'they do not replace the four-severity MAE/RMSE evaluation.']
    (diag/'telemetry_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir')
    parser.add_argument('--live-log')
    parser.add_argument('--resource-prefix')
    args=parser.parse_args()
    run_dir=Path(args.run_dir) if args.run_dir else resolve_run_dir(args.live_log) if args.live_log else None
    if run_dir is None:
        print('FARE_REPORT_SKIPPED reason=no_run_dir')
        return 0
    value=report(run_dir,args.resource_prefix)
    print(f'FARE_REPORT_PASS run_dir={run_dir} epochs={value["epoch_count"]}')
    return 0

if __name__=='__main__':raise SystemExit(main())
