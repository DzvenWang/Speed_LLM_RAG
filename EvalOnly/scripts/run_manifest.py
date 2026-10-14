#!/usr/bin/env python3
"""Run model evaluation wrappers sequentially in ONE allocated PBS GPU job.

Each completed item is marked atomically to permit resuming a manifest after a
walltime interruption. Failed items don't prevent later items from running.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def key(item):
    return hashlib.sha256(json.dumps(item, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def atomically_write(path, text):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(text, encoding='utf-8')
    temp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', required=True, choices=['backbone','baseline'])
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--dry-run', action='store_true', help='List subprocess commands, without running them')
    a = ap.parse_args()
    root = Path(a.root).expanduser().resolve()
    manifest = Path(a.manifest).expanduser().resolve()
    if not manifest.is_file(): ap.error(f'Manifest not found: {manifest}')
    jobs = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    if any(job.get('kind') != a.kind for job in jobs): ap.error('Manifest KIND mismatch')
    if not jobs: ap.error('Empty manifest')
    status_dir = manifest.parent / 'state' / manifest.stem
    if not a.dry_run: status_dir.mkdir(parents=True, exist_ok=True)
    failures, success, resumed = [], 0, 0
    print(f'EVAL_BATCH_START kind={a.kind} jobs={len(jobs)} allocated_gpus=1 manifest={manifest}', flush=True)
    for ix, item in enumerate(jobs, 1):
        signature = key(item)
        marker = status_dir / f'{ix:03d}_{signature}.done.json'
        label = item['label']
        if marker.exists():
            resumed += 1
            print(f'EVAL_BATCH_ALREADY_DONE index={ix}/{len(jobs)} label={label}', flush=True)
            continue
        if a.kind == 'backbone':
            target = Path(item['checkpoint'])
            env_updates = dict(MODEL=item['model'], TARGET=item['target'], MODE=item['mode'], SEED=str(item['seed']), CKPT=str(target))
            pbs = root / 'EvalOnly' / 'pbs' / 'backbone_eval_only.pbs'
        else:
            target = Path(item['formal_run_dir'])
            env_updates = dict(METHOD=item['method'], TARGET=item['target'], MODE=item['mode'], SEED=str(item['seed']), FORMAL_RUN_DIR=str(target))
            if item.get('backbone'): env_updates['BACKBONE'] = item['backbone']
            pbs = root / 'EvalOnly' / 'pbs' / 'baseline_eval_only.pbs'
        if a.dry_run:
            print(f'EVAL_BATCH_PREVIEW {ix}/{len(jobs)} {label} path={target} env={env_updates}', flush=True)
            continue
        if not pbs.is_file() or (not target.is_file() if a.kind == 'backbone' else not (target/'best.pt').is_file()):
            msg = f'Input or wrapper missing: {target} / {pbs}'
            failures.append((ix, label, msg))
            print(f'EVAL_BATCH_FAIL index={ix} label={label} error={msg}', flush=True)
            continue
        child_env = os.environ.copy()
        child_env.update(env_updates)
        child_env['SPEED_LLM_RAG_ROOT'] = str(root)
        child_env['PYTHONUNBUFFERED'] = '1'
        begin = time.monotonic()
        print(f'EVAL_BATCH_ITEM_START index={ix}/{len(jobs)} label={label} path={target}', flush=True)
        try:
            proc = subprocess.Popen(['bash', str(pbs)], env=child_env, cwd=root,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding='utf-8', errors='replace', bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line.rstrip('\n'), flush=True)
            rc = proc.wait()
        except Exception as exc:
            rc = 127
            print(f'EVAL_BATCH_CHILD_EXCEPTION index={ix} error={exc!r}', flush=True)
        elapsed = round(time.monotonic() - begin, 1)
        if rc == 0:
            success += 1
            payload = dict(label=label, index=ix, signature=signature, elapsed_seconds=elapsed,
                           completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), item=item, status='PASS')
            atomically_write(marker, json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
            print(f'EVAL_BATCH_ITEM_PASS index={ix}/{len(jobs)} label={label} seconds={elapsed}', flush=True)
        else:
            failures.append((ix, label, f'Exit code {rc}'))
            print(f'EVAL_BATCH_ITEM_FAIL index={ix}/{len(jobs)} label={label} rc={rc} seconds={elapsed}', flush=True)
    if not a.dry_run:
        report = dict(kind=a.kind, manifest=str(manifest), jobs=len(jobs), newly_passed=success,
                      previously_passed=resumed, failed=[dict(index=i,label=l,error=e) for i,l,e in failures],
                      ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        atomically_write(status_dir/'latest_summary.json', json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(f'EVAL_BATCH_END kind={a.kind} total={len(jobs)} passed_new={success} passed_previous={resumed} failed={len(failures)}', flush=True)
    for i, label, error in failures: print(f'EVAL_BATCH_FAILURE index={i} label={label} error={error}', flush=True)
    if failures: sys.exit(1)

if __name__ == '__main__': main()
