#!/usr/bin/env python3
"""Build/validate training and all frozen paired evaluation caches.

Evaluation tensors are pre-materialized even if Protocol access is still locked.
This is a preparation-only bypass: downstream model consumers must still respect
Protocol's evaluation-access gate before reading Test caches.
"""
from pathlib import Path
import argparse
import datetime as dt
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiment_common.utils import load_yaml
from experiment_common.cache import evaluation_access_granted

DEFAULT_MAIN = ROOT / 'configs' / 'experiment_contract.yaml'
EVAL_CONFIGS = [ROOT / 'configs' / f'experiment_contract_eval_s{s}.yaml' for s in (20,40,60,90)]

p = argparse.ArgumentParser()
p.add_argument('--force', action='store_true')
p.add_argument('--training-only', action='store_true', help='Build only development/formal caches; default builds evaluation too.')
p.add_argument('--log-file', default=None)
a = p.parse_args()

main_cfg = load_yaml(DEFAULT_MAIN)
log_root = Path(main_cfg['project'].get('log_root', REPO_ROOT / 'ExperimentCommon_logs'))
log_root.mkdir(parents=True, exist_ok=True)
stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
log_path = Path(a.log_file).resolve() if a.log_file else log_root / f'ExperimentCommonCacheBuild.{main_cfg["experiment"]["subgraph"]}.{stamp}.log'
log_f = log_path.open('w', encoding='utf-8', buffering=1)

def emit(*args):
    msg = ' '.join(str(x) for x in args)
    print(msg, flush=True)
    log_f.write(msg + '\n'); log_f.flush()

def run(args):
    cmd = [sys.executable] + [str(x) for x in args]
    emit('RUN', ' '.join(cmd))
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding='utf-8', errors='replace', bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip('\n')
        print(line, flush=True); log_f.write(line + '\n'); log_f.flush()
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)

def stage(label, args):
    emit(f'========== {label} START ==========')
    run(args)
    emit(f'========== {label} DONE ==========')

try:
    emit('LOG_FILE', log_path)
    emit('PROJECT_ROOT', REPO_ROOT)
    emit('EXPERIMENTCOMMON_ROOT', ROOT)
    emit('PYTHON', sys.executable)
    emit('START_TIME', dt.datetime.now().isoformat(timespec='seconds'))
    emit('MAIN_PROTOCOL', main_cfg['protocol']['name'])
    emit('MAIN_SEVERITY 0.40')
    emit('CACHE_POLICY dataset/protocol/severity/stage/full_stage_hash')
    emit('EVALUATION_POLICY DIRECT_FROZEN_PAIRED_CACHES')
    emit('EVALUATION_ACCESS DIRECT_UNLOCKED')

    stage('PREFLIGHT DEVELOPMENT', [ROOT/'scripts'/'preflight.py', '--stage', 'development'])
    stage('PREFLIGHT FORMAL', [ROOT/'scripts'/'preflight.py', '--stage', 'formal'])
    for st in ('development','formal'):
        cmd = [ROOT/'scripts'/'prepare_cache.py', '--stage', st]
        if a.force: cmd.append('--force')
        stage('PREPARE ' + st.upper(), cmd)
        stage('VALIDATE ' + st.upper(), [ROOT/'scripts'/'validate_cache.py', '--stage', st])
        stage('AUDIT SUPPORT ' + st.upper(), [ROOT/'scripts'/'audit_support_subsets.py', '--stage', st])

    if not a.training_only:
        for cfgp in EVAL_CONFIGS:
            sev = cfgp.stem.rsplit('_',1)[-1]
            stage('PREFLIGHT ' + sev.upper(), [ROOT/'scripts'/'preflight.py', '--config', cfgp, '--stage', 'evaluation'])
            cmd = [ROOT/'scripts'/'prepare_cache.py', '--config', cfgp, '--stage', 'evaluation', '--materialize-locked-evaluation']
            if a.force: cmd.append('--force')
            stage('PREPARE ' + sev.upper(), cmd)
            stage('VALIDATE ' + sev.upper(), [ROOT/'scripts'/'validate_cache.py', '--config', cfgp, '--stage', 'evaluation', '--validate-locked-evaluation'])
            stage('AUDIT SUPPORT ' + sev.upper(), [ROOT/'scripts'/'audit_support_subsets.py', '--config', cfgp, '--stage', 'evaluation'])
    else:
        emit('EVALUATION_CACHES SKIPPED (--training-only)')

    emit('EXPERIMENTCOMMON CACHE BUILD PASS')
    emit('END_TIME', dt.datetime.now().isoformat(timespec='seconds'))
except Exception as e:
    emit(f'EXPERIMENTCOMMON CACHE BUILD FAILED: {type(e).__name__}: {e}')
    emit('END_TIME', dt.datetime.now().isoformat(timespec='seconds'))
    raise
finally:
    log_f.close()
