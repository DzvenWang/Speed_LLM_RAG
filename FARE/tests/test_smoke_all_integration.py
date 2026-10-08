"""Subprocess-level smoke_all integration test, using fixture child commands.

This tests orchestration only; it is not an actual FARE/STLAformer run.
"""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile

RUNNER = Path(__file__).resolve().parents[1] / 'scripts' / 'smoke_all.py'

PRE_SCRIPT = '''import argparse,json
p=argparse.ArgumentParser()
p.add_argument('--dataset'); p.add_argument('--target'); p.add_argument('--config')
a=p.parse_args()
print('FARE_PREFLIGHT_PASS '+json.dumps({'dataset':a.dataset,'target':a.target}))
'''
TRAIN_SCRIPT = '''import argparse, json, os
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--dataset'); p.add_argument('--target'); p.add_argument('--config')
p.add_argument('--seed'); p.add_argument('--smoke',action='store_true'); p.add_argument('--run-dir')
a=p.parse_args()
if os.environ.get('FARE_TEST_FAIL_TARGET')==a.target:
  print('simulated training failure '+a.target)
  raise SystemExit(8)
run=Path(a.run_dir);run.mkdir(parents=True,exist_ok=True)
phases=['retrieval','recovery','injection','partial','full']
(run/'phase_summary.json').write_text(json.dumps([{'phase':s} for s in phases]))
for s in phases:
  (run/(s+'_best.pt')).write_bytes(b'test-fixture')
  d=run/'stages'/s;d.mkdir(parents=True,exist_ok=True)
  (d/'val_summary.json').write_text(json.dumps({'final':{'MAE':1.0,'RMSE':2.0,'directly_masked_MAE':1.5}}))
(run/'candidate_best.pt').write_bytes(b'test-fixture')
print('FARE_TRAIN_PASS run_dir='+str(run))
'''


def run_fixture(fail=None):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        scripts = root / 'FARE' / 'scripts'
        scripts.mkdir(parents=True)
        shutil.copy2(RUNNER, scripts / 'smoke_all.py')
        (scripts / 'preflight.py').write_text(PRE_SCRIPT)
        (scripts / 'train.py').write_text(TRAIN_SCRIPT)
        env = dict(__import__('os').environ)
        if fail:
            env['FARE_TEST_FAIL_TARGET'] = fail
        else:
            env.pop('FARE_TEST_FAIL_TARGET', None)
        cmd = [sys.executable, str(scripts / 'smoke_all.py'),
               '--dataset', 'TraffiDent_Net_01', '--seed', '10']
        result = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=20)
        log_dir = root / 'FARE_logs' / 'smoke' / 'TraffiDent_Net_01'
        reports = list(log_dir.glob('*/smoke_summary.json'))
        assert len(reports) == 1, reports
        summary = json.loads(reports[0].read_text())
        assert len(summary['targets']) == 2, summary
        assert 'speed' in summary['targets'] and 'flow' in summary['targets']
        if fail:
            assert result.returncode != 0
            assert summary['targets'][fail]['status'] == 'FAIL'
            assert summary['targets'][next(x for x in ('speed','flow') if x != fail)]['status'] == 'PASS'
            assert not summary['passed']
            assert 'FARE_SMOKE_ALL_FAIL' in result.stdout
        else:
            assert result.returncode == 0, (result.stdout, result.stderr)
            assert summary['targets']['speed']['status'] == 'PASS'
            assert summary['targets']['flow']['status'] == 'PASS'
            assert summary['passed']
            assert 'FARE_SMOKE_ALL_PASS' in result.stdout


if __name__ == '__main__':
    run_fixture()
    run_fixture('speed')
    run_fixture('flow')
    print('FARE_SMOKE_ALL_INTEGRATION_TEST_PASS (synthetic subprocess fixtures)')
