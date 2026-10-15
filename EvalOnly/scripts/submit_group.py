#!/usr/bin/env python3
"""One PBS job per project; each job processes all discovered models in series."""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
from catalog import discover


def main():
    parser = argparse.ArgumentParser(description='Submit one GPU job for Backbone, one for Baseline.')
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--dry-run', action='store_true', help='Preview both groups (default)')
    actions.add_argument('--submit-backbone', action='store_true')
    actions.add_argument('--submit-baseline', action='store_true')
    actions.add_argument('--submit-all', action='store_true')
    parser.add_argument('--root', default=os.environ.get('SPEED_LLM_RAG_ROOT'))
    parser.add_argument('--seed', type=int, default=int(os.environ.get('SEED', '10')))
    args = parser.parse_args()
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[2]
    for sub in ('Backbone', 'Baseline', 'Base_SupportBench'):
        if not (root / sub).exists(): parser.error(f'Missing {sub} in {root}')
    metric_py = root / 'Base_SupportBench' / 'experiment_common' / 'metrics.py'
    if not metric_py.is_file() or 'MAPE_PROTOCOL' not in metric_py.read_text():
        parser.error('MAPE Base_SupportBench is not installed')
    dry = not (args.submit_all or args.submit_backbone or args.submit_baseline)
    kinds = ['backbone', 'baseline'] if (dry or args.submit_all) else (
        ['backbone'] if args.submit_backbone else ['baseline'])
    staged = []
    for kind in kinds:
        jobs, skips = discover(root, kind, args.seed)
        print(f'\n======= {kind.upper()} ONE-GPU SEQUENTIAL EVAL =======', flush=True)
        for i, task in enumerate(jobs, 1):
            print(f'[{i:02d}] {task["label"]} => {task.get("checkpoint") or task.get("formal_run_dir")}', flush=True)
        for s in skips: print(f'[SKIP] {s} (no completed checkpoint)', flush=True)
        print(f'{kind}: {len(jobs)} evaluations; {len(skips)} skipped; PBS jobs: {int(bool(jobs))}', flush=True)
        if jobs: staged.append((kind, jobs))
    if dry:
        print(f'\nDRY RUN ONLY: {len(staged)} PBS jobs would be submitted; one GPU per job. No qsub executed.')
        return
    if not staged:
        print('Nothing to submit.')
        return
    run_root = root / 'EvalOnly' / 'runs' / (datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    run_root.mkdir(parents=True, exist_ok=False)
    pbs = root / 'EvalOnly' / 'pbs' / 'group_eval_only.pbs'
    if not pbs.is_file(): parser.error(f'Missing {pbs}')
    for kind, jobs in staged:
        manifest = run_root / f'{kind}.jsonl'
        with manifest.open('w', encoding='utf-8') as fh:
            for item in jobs: fh.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + '\n')
        vars = f'KIND={kind},MANIFEST={manifest},SPEED_LLM_RAG_ROOT={root}'
        envroot = os.environ.get('TFP_ENV_ROOT') or os.environ.get('CONDA_PREFIX')
        if envroot: vars += f',TFP_ENV_ROOT={Path(envroot).resolve()}'
        name = 'BB_Eval1GPU' if kind == 'backbone' else 'BL_Eval1GPU'
        cmd = ['qsub', '-N', name, '-v', vars, str(pbs)]
        print(f'SUBMIT {kind}: {len(jobs)} tasks, one GPU, manifest={manifest}', flush=True)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except OSError as ex:
            print(f'[ERROR] qsub not executable: {ex}', file=sys.stderr)
            raise SystemExit(1) from ex
        if proc.returncode:
            print(f'[ERROR] {kind} qsub rc={proc.returncode}: {proc.stderr}', file=sys.stderr)
            print(f'Resubmit: qsub -N {name} -v {vars} {pbs}', file=sys.stderr)
            raise SystemExit(proc.returncode)
        print(f'[SUBMITTED] {kind}: {proc.stdout.strip()}', flush=True)
    print(f'\nProgress manifests and task status: {run_root}')

if __name__ == '__main__': main()
