#!/usr/bin/env python3
"""Cross-platform runner for one explicit Backbone checkpoint on multiple severities.

Creates only the live combined .log under Speed_LLM_RAG/Backbone_logs.
All evaluation artifacts (JSON/CSV/predictions/summaries) are stored under Backbone_exps.
"""
from pathlib import Path
import argparse
import datetime
import os
import re
import subprocess
import sys
import json
from contextlib import nullcontext
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(os.environ.get('SPEED_LLM_RAG_ROOT', ROOT.parent)).expanduser().resolve()
sys.path.insert(0, str(ROOT))
from backbone.training_modes import mode_from_checkpoint, evaluation_mode_subdir
from backbone.evaluation_reporting import severity_table_lines, scenario_table_lines
from backbone.utils import load_yaml
_TS_RE = re.compile(r'^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]')


def _stamp_line(line: str) -> str:
    line = line.rstrip('\r\n')
    if _TS_RE.match(line):
        return line
    return f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {line}"


def _emit(fh, text: str) -> None:
    stamped = _stamp_line(text)
    print(stamped, flush=True)
    if fh is not None:
        fh.write(stamped + '\n')
        fh.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True, help='Backbone checkpoint path; absolute or relative to Speed_LLM_RAG root')
    ap.add_argument('--model', required=True, choices=['STAEformer','HimNet','BigST','STDN','PatchSTG','STLAformer'])
    ap.add_argument('--expected-seed', type=int, default=None)
    ap.add_argument('--severities', nargs='+', type=int, default=[20, 40, 60, 90], choices=[20, 40, 60, 90])
    ap.add_argument('--config', default=str(ROOT / 'configs' / 'backbone.yaml'))
    ap.add_argument('--run-stamp', default=None)
    ap.add_argument('--log-path', default=None,
                    help='Optional combined live-log path (used by GADI combined PBS)')
    ap.add_argument('--stdout-only', action='store_true',
                    help='Do not create a second evaluation live log; print only to parent stdout/stderr')
    ap.add_argument('--layout-snapshot', default=None)
    ap.add_argument('--allow-current-layout-mismatch', action='store_true')
    ap.add_argument('--smoke-test-batches', type=int, default=None,
                    help='Bound each severity to this many real test batches; skips full scenario-coverage assertion')
    args = ap.parse_args()

    stamp = args.run_stamp or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    seed_tag = str(args.expected_seed) if args.expected_seed is not None else 'unknown'
    log_root = REPO_ROOT / 'Backbone_logs'
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_path).expanduser().resolve() if args.log_path else (
        log_root / f'{args.model}.checkpoint_eval.all.seed{seed_tag}.{stamp}.live.log'
    )
    if not args.stdout_only:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(args.config)
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = (REPO_ROOT / checkpoint).resolve()
    if not checkpoint.is_file(): raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
    ckmeta=torch.load(checkpoint,map_location='cpu',weights_only=False)
    train_data_mode=mode_from_checkpoint(ckmeta)
    del ckmeta
    if args.log_path is None:
        log_path = log_root / f'{args.model}.checkpoint_eval.{train_data_mode}.all.seed{seed_tag}.{stamp}.live.log'
    checkpoint_run_tag = checkpoint.parent.name or 'checkpoint'
    result_root = (
        Path(cfg['project']['exp_root']) / args.model / cfg['experiment']['subgraph']
        / 'checkpoint_evaluation' / Path(*evaluation_mode_subdir(train_data_mode)) / checkpoint_run_tag / 'combined'
        / f'{stamp}_seed{seed_tag}'
    )
    result_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env.setdefault('SPEED_LLM_RAG_ROOT', str(REPO_ROOT))

    log_context = nullcontext(None) if args.stdout_only else log_path.open('a', encoding='utf-8', buffering=1)
    with log_context as fh:
        if args.stdout_only:
            _emit(fh, 'LOG_MODE=stdout_only_parent_managed')
        else:
            _emit(fh, f'LOG_FILE={log_path}')
        _emit(fh, f'START checkpoint evaluation checkpoint={args.checkpoint} severities={args.severities} expected_seed={args.expected_seed}')
        _emit(fh, 'EVALUATION_POLICY=simple_current_cache_current_layout')
        _emit(fh, f'TRAIN_DATA_MODE={train_data_mode} EVALUATION_INPUT_MODE=incomplete')

        severity_rows = []
        scenario_rows = []
        for sev in args.severities:
            result_json = result_root / f's{sev}.json'
            cmd = [
                sys.executable,
                str(ROOT / 'scripts' / 'evaluate_checkpoint.py'),
                '--checkpoint', args.checkpoint,
                '--model', args.model,
                '--severity', str(sev),
                '--config', args.config,
                '--run-stamp', stamp,
                '--result-json', str(result_json),
                '--quiet-summary',
            ]
            if args.expected_seed is not None:
                cmd += ['--expected-seed', str(args.expected_seed)]
            if args.layout_snapshot:
                cmd += ['--layout-snapshot', args.layout_snapshot]
            if args.allow_current_layout_mismatch:
                cmd += ['--allow-current-layout-mismatch']
            if args.smoke_test_batches is not None:
                cmd += ['--smoke-test-batches', str(args.smoke_test_batches)]
            _emit(fh, 'RUN ' + subprocess.list2cmdline(cmd))

            proc = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                _emit(fh, line)
            rc = proc.wait()
            _emit(fh, f'SEVERITY_EXIT_STATUS s{sev}={rc}')
            if rc != 0:
                _emit(fh, f'CHECKPOINT EVALUATION FAILED severity={sev} exit_status={rc}')
                raise SystemExit(rc)
            if not result_json.exists():
                _emit(fh, f'CHECKPOINT EVALUATION FAILED severity={sev}: result JSON missing: {result_json}')
                raise SystemExit(3)
            payload = json.loads(result_json.read_text(encoding='utf-8'))
            severity_rows.append(payload['severity'])
            scenario_rows.extend(payload['scenarios'])

        expected_families = [str(x) for x in cfg['experiment']['families']]
        if args.smoke_test_batches is None:
            for sev in args.severities:
                got = [str(r['scenario']) for r in scenario_rows if int(r['severity']) == int(sev)]
                if set(got) != set(expected_families) or len(got) != len(expected_families):
                    _emit(fh, f'SCENARIO_COVERAGE_FAILED s{sev} expected={expected_families} got={got}')
                    raise SystemExit(4)
            _emit(fh, 'SCENARIO_COVERAGE_PASS families=' + ','.join(expected_families))
        else:
            _emit(fh, f'SCENARIO_COVERAGE_SKIPPED smoke_test_batches={args.smoke_test_batches}')

        severity_csv = result_root / 'severity_summary.csv'
        scenario_csv = result_root / 'scenario_summary.csv'
        pd.DataFrame(severity_rows).to_csv(severity_csv, index=False)
        pd.DataFrame(scenario_rows).to_csv(scenario_csv, index=False)

        _emit(fh, '')
        _emit(fh, '=' * 118)
        _emit(fh, 'FINAL SEVERITY SUMMARY')
        _emit(fh, '=' * 118)
        for line in severity_table_lines(severity_rows): _emit(fh, line)
        _emit(fh, '')
        _emit(fh, '=' * 154)
        _emit(fh, 'FINAL SCENARIO SUMMARY')
        _emit(fh, 'Scenario = missing-sensing family; metrics are micro-aggregated within that scenario.')
        _emit(fh, '=' * 154)
        for line in scenario_table_lines(scenario_rows): _emit(fh, line)
        _emit(fh, '')
        _emit(fh, f'SEVERITY_SUMMARY_CSV={severity_csv}')
        _emit(fh, f'SCENARIO_SUMMARY_CSV={scenario_csv}')
        _emit(fh, 'CHECKPOINT EVALUATION PASS')
        if not args.stdout_only:
            _emit(fh, 'SCRIPT_EXIT_STATUS=0')


if __name__ == '__main__':
    main()
