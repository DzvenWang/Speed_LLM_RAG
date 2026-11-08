#!/usr/bin/env python3
"""Automatically run the v5.3.0 validation mechanism audit after training.

C12 four-switch: shared weights, common fixed gate (isolates C1/C2).
C3 eight-switch: shared controller-trained weights (isolates C1/C2/C3).
Negative controls: same C12 checkpoint, same first N validation batches.
All diagnostics are descriptive; no test labels are used to choose models.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES_C12 = ('M00', 'M10', 'M01', 'M11')
MODES_C3 = tuple(f'M{c}{s}{r}' for c in (0, 1) for s in (0, 1) for r in (0, 1))
NEGATIVES = (
    ('correct', 'correct'),  # the SAME limited validation subset
    ('shuffled', 'correct'),
    ('wrong_mask', 'correct'),
    ('correct', 'coverage'),
    ('correct', 'shuffled'),
)


@dataclass(frozen=True)
class DiagnosticJob:
    name: str
    checkpoint: Path
    modes: tuple[str, ...]
    controller_mode: str
    output_tag: str
    cf_control: str = 'correct'
    support_control: str = 'correct'
    max_batches: int = 0
    controller_variant: str | None = None
    disable_calibration: bool = False


def choose_c3_checkpoint(run: Path) -> Path | None:
    """Prefer the controller/joint checkpoint with the lowest recorded stage MAE."""
    summary = run / 'summary.json'
    candidates = []
    if summary.is_file():
        meta = json.loads(summary.read_text(encoding='utf-8'))
        for item in meta.get('stage_selection', []):
            if item.get('stage') not in ('controller', 'joint'):
                continue
            p = run / f"best_{item['stage']}.pt"
            if p.is_file():
                candidates.append((float(item['validation_mae']), str(p), p))
    if candidates:
        return min(candidates)[2]
    for stage in ('controller', 'joint'):
        p = run / f'best_{stage}.pt'
        if p.is_file():
            return p
    return None


def build_jobs(run: Path, primary_max_batches: int, negative_max_batches: int) -> list[DiagnosticJob]:
    c12 = run / 'best_c12.pt'
    if not c12.is_file():
        raise FileNotFoundError(f'Missing C12 checkpoint: {c12}')
    jobs = [DiagnosticJob('C12_four_switches', c12, MODES_C12, 'fixed',
                          'c12_four_switches', max_batches=primary_max_batches)]
    c3 = choose_c3_checkpoint(run)
    if c3 is not None:
        jobs.append(DiagnosticJob('C3_eight_switches', c3, MODES_C3, 'auto',
                                  'c3_eight_switches', max_batches=primary_max_batches))
    # Same trained C3 weights, with each branch trained by auxiliary
    # forecast loss. These are shared-weight mechanism diagnostics, not an
    # independent-retraining factorial study.
    if c3 is not None and negative_max_batches:
        for variant in ('scale','component','agreement','localized','full'):
            jobs.append(DiagnosticJob('C3_variant_'+variant,c3,('M111',),'auto',
                       'c3_variants',max_batches=negative_max_batches,
                       controller_variant=variant,disable_calibration=True))
    if negative_max_batches:
        for cf, support in NEGATIVES:
            jobs.append(DiagnosticJob(f'negative_{cf}_{support}', c12, ('M11',), 'fixed',
                                      'negative_controls', cf_control=cf,
                                      support_control=support, max_batches=negative_max_batches))
    return jobs


def result_dir(run: Path, job: DiagnosticJob) -> Path:
    return (run / 'evaluation' / 'validation' / 'automated' /
            job.output_tag / f'{job.cf_control}__{job.support_control}' /
            ('crsi_'+job.controller_variant if job.controller_variant else ''))


def command(job: DiagnosticJob, batch_size: int) -> list[str]:
    cmd = [sys.executable, '-u', str(ROOT / 'scripts' / 'evaluate_v5.py'),
           '--checkpoint', str(job.checkpoint), '--split', 'validation',
           '--validation-section','check',
           '--controller-mode', job.controller_mode, '--modes', *job.modes,
           '--batch-size', str(batch_size), '--cf-control', job.cf_control,
           '--support-control', job.support_control,
           '--output-tag', job.output_tag, '--diagnostics']
    if job.controller_variant:cmd += ['--controller-variant',job.controller_variant]
    if job.disable_calibration:cmd += ['--disable-calibration']
    if job.max_batches:
        cmd += ['--max-batches', str(job.max_batches)]
    return cmd


def read_mode_row(run: Path, job: DiagnosticJob, mode: str) -> dict:
    table = result_dir(run, job) / 'mechanism_v5.csv'
    if not table.is_file():
        raise FileNotFoundError(f'Missing diagnostic CSV: {table}')
    with table.open(encoding='utf-8', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['mode'] == mode]
    if len(rows) != 1:
        raise RuntimeError(f'Expected exactly one row for {mode} in {table}, got {len(rows)}')
    return rows[0]


def summarize(run: Path, jobs: list[DiagnosticJob]) -> dict:
    """Aggregate comparisons without mixing full validation and limited subsets."""
    by_name = {j.name: j for j in jobs}
    main = {}
    for name, mode in (('C12_four_switches', 'M11'), ('C3_eight_switches', 'M111')):
        if name in by_name:
            row = read_mode_row(run, by_name[name], mode)
            main[name] = dict(checkpoint=str(by_name[name].checkpoint),
                              full_mode=mode, MAE=float(row['MAE']), RMSE=float(row['RMSE']),
                              evaluated_queries=int(row['evaluated_queries']),
                              full_split_queries=int(row['full_split_queries']),
                              csv=str(result_dir(run, by_name[name]) / 'mechanism_v5.csv'))
    comparisons = []
    base_name = 'negative_correct_correct'
    if base_name in by_name:
        reference = read_mode_row(run, by_name[base_name], 'M11')
        for name in (j.name for j in jobs if j.name.startswith('negative_') and j.name != base_name):
            j = by_name[name]
            row = read_mode_row(run, j, 'M11')
            if (row['evaluated_queries'] != reference['evaluated_queries'] or
                    row['full_split_queries'] != reference['full_split_queries'] or
                    row['sample_index_sha256'] != reference['sample_index_sha256']):
                raise RuntimeError('Negative controls must use exactly the same sample budget')
            if j.cf_control=='wrong_mask' and float(row.get('wrong_mask_effective_rate') or 0)<0.99:
                raise RuntimeError('Wrong-mask comparison has insufficient distinct-mask coverage')
            comparisons.append(dict(cf_control=j.cf_control,
                                    support_control=j.support_control,
                                    sample_queries=int(row['evaluated_queries']),
                                    reference_MAE=float(reference['MAE']),
                                    control_MAE=float(row['MAE']),
                                    MAE_gap_control_minus_correct=float(row['MAE'])-float(reference['MAE']),
                                    reference_RMSE=float(reference['RMSE']),
                                    control_RMSE=float(row['RMSE']),
                                    RMSE_gap_control_minus_correct=float(row['RMSE'])-float(reference['RMSE'])))
    c3_comparisons=[]
    variant_names=[j.name for j in jobs if j.name.startswith('C3_variant_')]
    if variant_names:
        reference=read_mode_row(run,by_name['C3_variant_full'],'M111')
        for name in variant_names:
            row=read_mode_row(run,by_name[name],'M111')
            if (row['sample_index_sha256']!=reference['sample_index_sha256'] or
                row['evaluated_queries']!=reference['evaluated_queries']):
                raise RuntimeError('C3 variants must use exactly the same validation queries')
            c3_comparisons.append(dict(variant=by_name[name].controller_variant,
                    MAE=float(row['MAE']),RMSE=float(row['RMSE']),
                    delta_MAE_against_full=float(row['MAE'])-float(reference['MAE']),
                    evaluated_queries=int(row['evaluated_queries'])))
    result = dict(version='5.3.0', split='chronological_check',
                  note='C12 negative controls use a deterministic matched subset; do not compare their absolute MAE to full-validation results.',
                  primary_results=main, negative_controls=comparisons,
                  c3_variant_comparisons=c3_comparisons,
                  completed_at=dt.datetime.now(dt.timezone.utc).isoformat())
    folder = run / 'evaluation' / 'validation' / 'automated'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'diagnostic_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if c3_comparisons:
        with (folder/'c3_variant_comparison.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(c3_comparisons[0]))
            writer.writeheader();writer.writerows(c3_comparisons)
    if comparisons:
        with (folder / 'negative_control_comparison.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
    return result


def run_jobs(run: Path, jobs: list[DiagnosticJob], batch_size: int) -> dict:
    folder = run / 'evaluation' / 'validation' / 'automated'
    folder.mkdir(parents=True, exist_ok=True)
    events = []
    journal = folder / 'automatic_diagnosis.log'
    status_file = folder / 'job_status.json'
    for i, job in enumerate(jobs, 1):
        cmd = command(job, batch_size)
        print(f'V5_DIAGNOSTIC_START {i}/{len(jobs)} name={job.name} '
              f'checkpoint={job.checkpoint} max_batches={job.max_batches}', flush=True)
        with journal.open('a', encoding='utf-8') as log:
            log.write(f'\n===== {job.name} =====\nCOMMAND: {" ".join(cmd)}\n')
            log.flush()
            process = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
            code = process.wait()
        event = dict(name=job.name, success=(code == 0), returncode=code,
                     checkpoint=str(job.checkpoint), modes=job.modes,
                     cf_control=job.cf_control, support_control=job.support_control,
                     max_batches=job.max_batches, result_path=str(result_dir(run, job)))
        events.append(event)
        status_file.write_text(json.dumps(events, indent=2), encoding='utf-8')
        if code:
            print(f'V5_DIAGNOSTIC_FAILED name={job.name} code={code}', flush=True)
            raise subprocess.CalledProcessError(code, cmd)
        print(f'V5_DIAGNOSTIC_PASS name={job.name}', flush=True)
    summary = summarize(run, jobs)
    print('V5_AUTO_DIAGNOSIS_PASS summary='+str(folder / 'diagnostic_summary.json'), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--negative-max-batches', type=int, default=64)
    parser.add_argument('--primary-max-batches', type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.negative_max_batches < 0 or args.primary_max_batches < 0:
        parser.error('Batch size must be positive and max-batches limits nonnegative')
    run = Path(args.run_dir).resolve()
    if not run.is_dir():
        parser.error(f'Missing run directory: {run}')
    jobs = build_jobs(run, args.primary_max_batches, args.negative_max_batches)
    print(f'V5_DIAGNOSTIC_PLAN jobs={len(jobs)} '+
          ' '.join(f'{j.name}:{j.max_batches or "full"}' for j in jobs), flush=True)
    run_jobs(run, jobs, args.batch_size)


if __name__ == '__main__':
    main()
