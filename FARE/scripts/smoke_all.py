#!/usr/bin/env python3
"""Run FARE preflight and full five-phase smoke for speed and flow.

Targets are processed sequentially (safe for one GPU). A failed target does
not prevent the other target from being checked. Exit status is nonzero unless
*every* requested target passes its preflight, training and artifact checks.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

PHASES = ('retrieval', 'recovery', 'injection', 'partial', 'full')
SUPPORTED_DATASETS = tuple(f'TraffiDent_Net_{n:02d}' for n in (1, 2, 3, 4, 6))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='TraffiDent_Net_06', choices=SUPPORTED_DATASETS)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--targets', nargs='+', choices=('speed', 'flow'), default=['speed', 'flow'])
    parser.add_argument('--config', help='FARE YAML configuration shared by both targets')
    parser.add_argument('--log-root', type=Path, help='Root for the combined smoke logs and summary')
    parser.add_argument('--skip-preflight', action='store_true')
    return parser.parse_args(argv)


def run_child(cmd, log_file: Path, prefix: str):
    """Run a subprocess with streamed output saved to a target-specific log."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env.setdefault('SPEED_LLM_RAG_ROOT', str(Path(__file__).resolve().parents[2]))
    found_preflight_pass = False
    found_training_pass = False
    started = time.monotonic()
    with log_file.open('w', encoding='utf-8') as logfile:
        logfile.write('COMMAND: ' + subprocess.list2cmdline([str(x) for x in cmd]) + '\n')
        logfile.flush()
        child = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
        )
        assert child.stdout is not None
        try:
            for line in child.stdout:
                line = line.rstrip('\r\n')
                logfile.write(line + '\n')
                logfile.flush()
                print(f'[{prefix}] {line}', flush=True)
                found_preflight_pass |= line.startswith('FARE_PREFLIGHT_PASS ')
                found_training_pass |= line.startswith('FARE_TRAIN_PASS ')
        except KeyboardInterrupt:
            child.terminate()
            child.wait()
            raise
        exit_code = child.wait()
    return {
        'exit_code': exit_code,
        'seconds': round(time.monotonic() - started, 2),
        'preflight_marker': found_preflight_pass,
        'train_marker': found_training_pass,
        'log': str(log_file.resolve()),
    }


def validate_smoke_run(run_dir: Path):
    """Ensure every phase was both trained and independently evaluated."""
    problems = []
    summary = run_dir / 'phase_summary.json'
    if not summary.is_file():
        return {'ok': False, 'phases': [], 'phase_metrics': {},
                'problems': ['Missing phase_summary.json']}
    try:
        records = json.loads(summary.read_text(encoding='utf-8'))
    except (ValueError, OSError) as exc:
        return {'ok': False, 'phases': [], 'phase_metrics': {},
                'problems': [f'Invalid phase_summary.json: {exc}']}
    if not isinstance(records, list):
        return {'ok': False, 'phases': [], 'phase_metrics': {},
                'problems': ['Expected phase_summary.json to contain a list']}
    phases = [record.get('phase') for record in records if isinstance(record, dict)]
    if phases != list(PHASES):
        problems.append(f'Expected phases {list(PHASES)}, got {phases}')
    metrics = {}
    for phase in PHASES:
        ckpt = run_dir / f'{phase}_best.pt'
        stage_report = run_dir / 'stages' / phase / 'val_summary.json'
        if not ckpt.is_file():
            problems.append(f'Missing checkpoint: {ckpt.name}')
        if not stage_report.is_file():
            problems.append(f'Missing stage evaluation: stages/{phase}/val_summary.json')
            continue
        try:
            obj = json.loads(stage_report.read_text(encoding='utf-8'))
            final = obj['final']
            metrics[phase] = {
                'MAE': float(final['MAE']),
                'RMSE': float(final['RMSE']),
                'directly_masked_MAE': float(final['directly_masked_MAE']),
            }
            if not all(math.isfinite(x) for x in metrics[phase].values()):
                problems.append(f'Nonfinite evaluation metric at {phase}')
        except (ValueError, OSError, TypeError, KeyError) as exc:
            problems.append(f'Invalid evaluation report at {phase}: {exc}')
    if not (run_dir / 'candidate_best.pt').is_file():
        problems.append('Missing candidate_best.pt')
    return {'ok': not problems, 'phases': phases, 'phase_metrics': metrics, 'problems': problems}


def main(argv=None):
    args = parse_args(argv)
    if len(set(args.targets)) != len(args.targets):
        raise SystemExit('Duplicate targets not allowed')
    source_root = Path(__file__).resolve().parents[2]
    scripts = Path(__file__).resolve().parent
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log_root = args.log_root or source_root / 'FARE_logs' / 'smoke' / args.dataset / f'{stamp}_seed{args.seed}'
    log_root.mkdir(parents=True, exist_ok=True)
    summary_path = log_root / 'smoke_summary.json'
    report = {'dataset': args.dataset, 'seed': args.seed, 'started_at': datetime.now().isoformat(),
              'requested_targets': args.targets, 'targets': {}, 'passed': False}

    for target in args.targets:
        print(f'\n========== FARE {args.dataset} / {target.upper()} SMOKE ==========', flush=True)
        run_dir = source_root / 'FARE_exps' / target / args.dataset / 'formal_smoke' / f'{stamp}_seed{args.seed}'
        result = {'status': 'FAIL', 'run_dir': str(run_dir), 'preflight': None, 'training': None,
                  'validation': None, 'problems': []}
        common = ['--dataset', args.dataset, '--target', target]
        if args.config:
            common += ['--config', args.config]
        if not args.skip_preflight:
            preflight = run_child(
                [sys.executable, scripts / 'preflight.py', *common],
                log_root / f'{target}_preflight.log', f'{target}:preflight'
            )
            result['preflight'] = preflight
            if preflight['exit_code'] != 0 or not preflight['preflight_marker']:
                result['problems'].append('Preflight did not complete successfully')
        if not result['problems']:
            training = run_child(
                [sys.executable, scripts / 'train.py', *common, '--seed', str(args.seed),
                 '--smoke', '--run-dir', str(run_dir)],
                log_root / f'{target}_train.log', f'{target}:train'
            )
            result['training'] = training
            if training['exit_code'] != 0 or not training['train_marker']:
                result['problems'].append('Smoke training failed or FARE_TRAIN_PASS was absent')
            result['validation'] = validate_smoke_run(run_dir)
            result['problems'].extend(result['validation']['problems'])
        if not result['problems']:
            result['status'] = 'PASS'
        report['targets'][target] = result
        report['passed'] = all(item['status'] == 'PASS' for item in report['targets'].values()) and len(report['targets']) == len(args.targets)
        summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'FARE_SMOKE_TARGET {target} {result["status"]}', flush=True)
        if result['problems']:
            print('  ' + '; '.join(result['problems']), flush=True)

    report['finished_at'] = datetime.now().isoformat()
    report['passed'] = len(report['targets']) == len(args.targets) and all(
        x['status'] == 'PASS' for x in report['targets'].values())
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('FARE_SMOKE_SUMMARY ' + json.dumps({
        'dataset': args.dataset, 'seed': args.seed,
        'targets': {k: v['status'] for k, v in report['targets'].items()},
        'summary_path': str(summary_path),
    }), flush=True)
    print('FARE_SMOKE_ALL_PASS' if report['passed'] else 'FARE_SMOKE_ALL_FAIL', flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
