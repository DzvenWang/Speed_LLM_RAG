"""Two-job PBS workflow and combined postprocess protocol regression."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SUBMIT = ROOT / 'pbs/submit_pipeline_v5.sh'
POST = ROOT / 'pbs/postprocess_v5.pbs'


def mocked_qsub(tmp_path, **overrides):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    qsub = bin_dir / 'qsub'
    qsub.write_text('''#!/usr/bin/env python3
import os, sys, json
p = os.environ['QSUB_TRACE']
with open(p, 'a') as fh:
    i = sum(1 for _ in open(p)) if os.path.getsize(p) else 0
    fh.write(json.dumps(sys.argv[1:]) + '\\n')
print(f'18246000{i+1}.gadi-pbs')
''')
    qsub.chmod(0o755)
    ckpt = tmp_path / 'best.pt'
    ckpt.write_bytes(b'ckpt')
    trace = tmp_path / 'calls.jsonl'
    trace.write_text('')
    env = os.environ.copy()
    for key in ('BATCH_SIZE','SMOKE','SMOKE_PROFILE_FULL_BATCH','TCRR_PBS_MEM',
                'TCRR_POST_PBS_MEM', 'TCRR_POST_WALLTIME','TCRR_PBS_NCPUS',
                'TCRR_TRAIN_WORKERS','DIAGNOSTIC_BATCH_SIZE'):
        env.pop(key, None)
    env.update({
        'PATH': str(bin_dir) + os.pathsep + env.get('PATH', ''),
        'QSUB_TRACE': str(trace), 'TARGET': 'speed', 'SEED': '10',
        'RUN_STAMP': '20990808_123456', 'SPEED_LLM_RAG_ROOT': str(REPO),
        'BACKBONE_CHECKPOINT': str(ckpt),
    })
    env.update({k: str(v) for k,v in overrides.items()})
    result = subprocess.run(['bash', str(SUBMIT)], cwd=REPO,
                            env=env, capture_output=True, text=True)
    calls = [json.loads(line) for line in trace.read_text().splitlines()]
    return result, calls


def _vars(call):
    return dict(part.split('=',1) for part in call[call.index('-v') + 1].split(','))


def test_default_pipeline_creates_only_train_and_post(tmp_path):
    result, calls = mocked_qsub(tmp_path)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 2
    assert calls[0][-1] == 'TCRR-STLA/pbs/formal_v5.pbs'
    assert calls[1][-1] == 'TCRR-STLA/pbs/postprocess_v5.pbs'
    assert 'mem=96GB' in calls[0] and 'mem=32GB' in calls[1]
    assert 'depend=afterok:182460001.gadi-pbs' in calls[1]
    assert 'TRAIN_ID=182460001.gadi-pbs' in result.stdout
    assert 'POSTPROCESS_ID=182460002.gadi-pbs' in result.stdout
    assert 'POST_PBS_MEM=32GB' in result.stdout
    assert 'DIAGNOSTICS_ID=' not in result.stdout and 'TEST_ID=' not in result.stdout
    assert _vars(calls[0])['RUN_DIR'] == _vars(calls[1])['RUN_DIR']
    assert 'SKIP_DIAGNOSTICS' not in _vars(calls[1])
    assert 'SKIP_EVALUATE' not in _vars(calls[1])


def test_post_memory_and_walltime_overrides_are_independent(tmp_path):
    result,calls=mocked_qsub(tmp_path,TCRR_PBS_MEM='64GB',
                             TCRR_POST_PBS_MEM='48GB',TCRR_POST_WALLTIME='24:00:00')
    assert result.returncode==0,result.stderr
    assert 'mem=64GB' in calls[0]
    assert 'mem=48GB' in calls[1]
    assert 'walltime=24:00:00' in calls[1]
    assert 'POST_PBS_MEM=48GB' in result.stdout


@pytest.mark.parametrize('override', [
    {'TCRR_POST_PBS_MEM':'0GB'},
    {'TCRR_POST_PBS_MEM':'512GB'},
    {'TCRR_POST_PBS_MEM':'bad'},
    {'TCRR_POST_WALLTIME':'nonsense'},
    {'TCRR_POST_WALLTIME':'00:00:00'},
])
def test_bad_post_configuration_rejected_before_qsub(tmp_path, override):
    result,calls=mocked_qsub(tmp_path,**override)
    assert result.returncode!=0
    assert not calls


def test_smoke_still_only_one_job(tmp_path):
    result,calls=mocked_qsub(tmp_path,SMOKE=1)
    assert result.returncode==0,result.stderr
    assert len(calls)==1
    assert calls[0][-1]=='TCRR-STLA/pbs/formal_v5.pbs'


def test_combined_post_does_not_run_test_when_diagnostics_fails(tmp_path):
    job = tmp_path / 'post.pbs'
    # Replace PBS runtime setup with a fake test Python launcher.
    src = POST.read_text()
    src = src.replace('source "${PBS_O_WORKDIR:?}/TCRR-STLA/pbs/runtime.sh"',
                      'PYTHON="${FAKE_PYTHON:?}"')
    job.write_text(src)
    fake_python = tmp_path / 'python'
    fake_python.write_text('''#!/usr/bin/env python3
import os,sys
with open(os.environ['CALLS'], 'a') as fh:
    fh.write(' '.join(sys.argv[1:]) + '\\n')
if 'auto_diagnose_v5.py' in ' '.join(sys.argv) and os.environ.get('FAIL_DIAG') == '1':
    sys.exit(31)
''')
    fake_python.chmod(0o755)
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    (run_dir / 'best_c12.pt').write_bytes(b'ckpt')
    (run_dir / 'best_controller.pt').write_bytes(b'ckpt')
    (run_dir / 'summary.json').write_text('{"best_stage":"controller"}')
    (run_dir / 'validation_partitions.json').write_text('{"selection":[],"calibration":[],"check":[]}')
    calls = tmp_path / 'calls.txt'
    for fail_diag in ('1', '0'):
        calls.write_text('')
        env = os.environ.copy()
        env.update(RUN_DIR=str(run_dir),FAKE_PYTHON=str(fake_python),
                   CALLS=str(calls),FAIL_DIAG=fail_diag,PBS_O_WORKDIR=str(REPO))
        result = subprocess.run(['bash', str(job)], cwd=REPO,
                                env=env,capture_output=True,text=True)
        records = calls.read_text().splitlines()
        assert 'auto_diagnose_v5.py' in records[0]
        if fail_diag=='1':
            assert result.returncode != 0
            assert len(records)==1  # No test after failed diagnostics
            assert 'V53_TEST_PASS' not in result.stdout
        else:
            assert result.returncode == 0, result.stderr
            assert len(records)==3  # diagnostics, C12 test, controller test
            assert 'V53_POSTPROCESS_PASS' in result.stdout
            assert 'V53_DIAGNOSTICS_PASS' in result.stdout
            assert 'V53_TEST_PASS' in result.stdout
            assert 'evaluate_v5.py' in records[1]
    for name in ('diagnostics.live.log','c12_test.live.log','controller_test.live.log'):
        assert (run_dir / 'postprocess_logs' / name).is_file()


def test_replacement_post_submission_preserves_original_train(tmp_path):
    bin_dir=tmp_path/'bin';bin_dir.mkdir()
    qstat=bin_dir/'qstat'
    run_dir=str(REPO/'TCRR-STLA_exps/speed/incomplete/TraffiDent_Net_06/v5/formal/20991231_222222_seed10_full')
    qstat.write_text('#!/usr/bin/env python3\nimport json\n'
                     f'print(json.dumps({{"Jobs": {{"182459670.gadi-pbs": '
                     f'{{"Variable_List": {{"RUN_DIR": {run_dir!r}}}}}}}}}))\n')
    qstat.chmod(0o755)
    qsub=bin_dir/'qsub'
    trace=tmp_path/'replacement.jsonl'
    qsub.write_text('#!/usr/bin/env python3\nimport sys,os,json\n'
                    'with open(os.environ["TRACE"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n'
                    'print("182460999.gadi-pbs")\n')
    qsub.chmod(0o755)
    env=os.environ.copy()
    env.update(PATH=str(bin_dir)+os.pathsep+env.get('PATH',''),TRACE=str(trace),
               SPEED_LLM_RAG_ROOT=str(REPO))
    env.pop('RUN_DIR',None)
    job=ROOT/'pbs/submit_post_for_existing_train_v5.sh'
    out=subprocess.run(['bash',str(job),'182459670.gadi-pbs'],cwd=REPO,
                       env=env,capture_output=True,text=True)
    assert out.returncode==0,out.stderr
    assert 'PRESERVED_TRAIN_ID=182459670.gadi-pbs' in out.stdout
    assert 'POSTPROCESS_ID=182460999.gadi-pbs' in out.stdout
    calls=[json.loads(x) for x in trace.read_text().splitlines()]
    assert len(calls)==1
    assert 'depend=afterok:182459670.gadi-pbs' in calls[0]
    assert 'mem=32GB' in calls[0]
    assert _vars(calls[0])['RUN_DIR']==run_dir


def test_replacement_post_fails_closed_when_qstat_unavailable(tmp_path):
    bin_dir=tmp_path/'bin';bin_dir.mkdir()
    qstat=bin_dir/'qstat';qstat.write_text('#!/bin/sh\nexit 7\n');qstat.chmod(0o755)
    trace=tmp_path/'called'
    qsub=bin_dir/'qsub';qsub.write_text(f'#!/bin/sh\ntouch "{trace}"\n');qsub.chmod(0o755)
    env=os.environ.copy()
    env.update(PATH=str(bin_dir)+os.pathsep+env.get('PATH',''),SPEED_LLM_RAG_ROOT=str(REPO))
    env.pop('RUN_DIR',None)
    result=subprocess.run(['bash',str(ROOT/'pbs/submit_post_for_existing_train_v5.sh'),
                           '182459670.gadi-pbs'],env=env,cwd=REPO,capture_output=True,text=True)
    assert result.returncode!=0
    assert not trace.exists()
