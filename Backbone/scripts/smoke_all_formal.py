#!/usr/bin/env python3
"""Run all Backbone models x targets through real Formal train + evaluation smoke tests."""
from pathlib import Path
import argparse, datetime, os, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[1]
ORDER = ('STAEformer','HimNet','BigST','STDN','PatchSTG','STLAformer')
TARGETS = ('speed','flow')
SEVERITIES = (20,40,60,90)


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)
    def flush(self):
        for stream in self.streams:
            stream.flush()


def run_streamed(cmd, *, cwd, env):
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end='', flush=True)
    return proc.wait()


def main():
    ap = argparse.ArgumentParser(description='All-model real-Formal smoke: bounded training + s20/s40/s60/s90 evaluation.')
    ap.add_argument('--seed', type=int, default=10)
    ap.add_argument('--train-batches', type=int, default=2)
    ap.add_argument('--val-batches', type=int, default=1)
    ap.add_argument('--test-batches', type=int, default=1)
    ap.add_argument('--continue-on-error', action='store_true')
    ap.add_argument('--training-only', action='store_true', help='Retain the old training-only smoke behaviour')
    a = ap.parse_args()
    if a.seed != 10:
        raise SystemExit('Formal smoke seed is fixed to 10')
    if min(a.train_batches, a.val_batches, a.test_batches) < 1:
        raise SystemExit('Smoke batch counts must all be >= 1')

    log_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = ROOT.parent / 'Backbone_logs' / 'smoke_all_formal'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'BackboneSmokeAllFormal.{log_stamp}.log'
    log_fp = log_path.open('w', encoding='utf-8', buffering=1)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(original_stdout, log_fp)
    sys.stderr = Tee(original_stderr, log_fp)

    jobs = [(m,t) for m in ORDER for t in TARGETS]
    failures = []
    t0 = time.time()
    print(f'ALL_FORMAL_SMOKE_LOG path={log_path}', flush=True)
    print(f'ALL_FORMAL_SMOKE_START total={len(jobs)} seed={a.seed} training_only={a.training_only}', flush=True)
    for i,(model,target) in enumerate(jobs,1):
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        print(f'[{i:02d}/{len(jobs)}] START model={model} target={target}', flush=True)
        env = os.environ.copy()
        env['TRAFFIDENT_TARGET'] = target
        env['PYTHONUNBUFFERED'] = '1'
        env.setdefault('SPEED_LLM_RAG_ROOT', str(ROOT.parent))

        train_cmd = [sys.executable,'-u',str(ROOT/'scripts/run_model_formal.py'),'--model',model,'--target',target,'--seed',str(a.seed),'--run-stamp',stamp,'--smoke','--smoke-train-batches',str(a.train_batches),'--smoke-val-batches',str(a.val_batches)]

        rc = run_streamed(train_cmd, cwd=ROOT, env=env)
        if rc != 0:
            failures.append((model,target,'train',rc))
            print(f'[{i:02d}/{len(jobs)}] FAIL TRAIN model={model} target={target} rc={rc}', flush=True)
            if not a.continue_on_error: break
            continue
        print(f'[{i:02d}/{len(jobs)}] TRAIN_PASS model={model} target={target}', flush=True)

        if a.training_only:
            continue

        checkpoint = ROOT.parent/'Backbone_exps'/target/model/'TraffiDent_Net_06'/'formal_smoke'/f'{stamp}_seed{a.seed}'/'smoke.pt'
        if not checkpoint.exists():
            failures.append((model,target,'checkpoint_missing',2))
            print(f'[{i:02d}/{len(jobs)}] FAIL checkpoint missing: {checkpoint}', flush=True)
            if not a.continue_on_error: break
            continue

        eval_cmd = [
            sys.executable,'-u',str(ROOT/'scripts/run_checkpoint_evaluations.py'),
            '--checkpoint',str(checkpoint),'--model',model,'--expected-seed',str(a.seed),
            '--severities',*[str(x) for x in SEVERITIES],
            '--run-stamp',stamp,'--stdout-only','--smoke-test-batches',str(a.test_batches),
        ]
        rc = run_streamed(eval_cmd, cwd=ROOT, env=env)
        if rc != 0:
            failures.append((model,target,'evaluation',rc))
            print(f'[{i:02d}/{len(jobs)}] FAIL EVAL model={model} target={target} rc={rc}', flush=True)
            if not a.continue_on_error: break
            continue
        print(f'[{i:02d}/{len(jobs)}] FULL_PASS model={model} target={target} severities=20,40,60,90', flush=True)

    if failures:
        detail = ','.join(f'{m}:{t}:{stage}:rc{r}' for m,t,stage,r in failures)
        print('ALL_FORMAL_SMOKE_FAIL failures='+detail, flush=True)
        raise SystemExit(1)
    mode = 'training_only' if a.training_only else 'train_plus_eval'
    print(f'ALL_FORMAL_SMOKE_PASS {len(jobs)}/{len(jobs)} mode={mode} seconds={time.time()-t0:.1f}', flush=True)


if __name__ == '__main__':
    main()
