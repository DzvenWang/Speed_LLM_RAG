"""Default launch profile vs explicit overrides without invoking real qsub."""
from __future__ import annotations
import json, os, subprocess
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent
SCRIPT=ROOT/'pbs'/'submit_pipeline_v5.sh'


def mock_submit(tmp_path, **overrides):
    b=tmp_path/'bin';b.mkdir()
    qsub=b/'qsub'
    qsub.write_text('#!/usr/bin/env python3\nimport os,sys,json\n'
                    'with open(os.environ["FAKE_QSUB_LOG"],"a") as f:'
                    'f.write(json.dumps(sys.argv[1:])+"\\n")\n'
                    'print("184000001.gadi-pbs")\n')
    qsub.chmod(0o755)
    ckpt=tmp_path/'best.pt';ckpt.write_bytes(b'ckpt')
    log=tmp_path/'qsub.jsonl'
    env=os.environ.copy()
    for key in ('BATCH_SIZE','TCRR_PBS_NCPUS','TCRR_PBS_MEM','TCRR_TRAIN_WORKERS',
                'TCRR_VAL_WORKERS','TCRR_EVAL_WORKERS','TCRR_PERSISTENT_WORKERS',
                'TCRR_PREFETCH_FACTOR','SMOKE','SMOKE_PROFILE_FULL_BATCH',
                'DIAGNOSTIC_BATCH_SIZE','DIAGNOSTIC_MAX_BATCHES','DIAGNOSTIC_PRIMARY_MAX_BATCHES'):
        env.pop(key,None)
    env.update({'PATH':str(b)+os.pathsep+env.get('PATH',''),
                'FAKE_QSUB_LOG':str(log),'SPEED_LLM_RAG_ROOT':str(REPO),
                'RUN_STAMP':'20991111_000000','TARGET':'speed',
                'BACKBONE_CHECKPOINT':str(ckpt)})
    env.update({k:str(v) for k,v in overrides.items()})
    cp=subprocess.run(['bash',str(SCRIPT)],cwd=REPO,env=env,text=True,capture_output=True)
    calls=[json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return cp,calls


def params(call):
    return dict(piece.split('=',1) for piece in call[call.index('-v')+1].split(','))


def test_default_config_transmitted_both_jobs(tmp_path):
    cp,calls=mock_submit(tmp_path)
    assert cp.returncode == 0, cp.stderr
    assert len(calls)==2
    for call in calls:
        assert 'ncpus=12' in call
        assert ('mem=96GB' in call) if call is calls[0] else ('mem=32GB' in call)
        v=params(call)
        assert {k:v[k] for k in (
            'BATCH_SIZE','TCRR_TRAIN_WORKERS','TCRR_VAL_WORKERS',
            'TCRR_PERSISTENT_WORKERS','TCRR_PREFETCH_FACTOR')} == {
            'BATCH_SIZE':'64','TCRR_TRAIN_WORKERS':'4','TCRR_VAL_WORKERS':'0',
            'TCRR_PERSISTENT_WORKERS':'1','TCRR_PREFETCH_FACTOR':'2'}
    assert 'BATCH_SIZE=64' in cp.stdout
    assert 'TRAIN_WORKERS=4' in cp.stdout
    assert 'TRAIN_PBS_MEM=96GB' in cp.stdout
    assert 'POST_PBS_MEM=32GB' in cp.stdout


def test_explicit_overrides_including_zero_take_precedence(tmp_path):
    cp,calls=mock_submit(tmp_path,BATCH_SIZE=32,TCRR_PBS_MEM='32GB',
                         TCRR_PBS_NCPUS=12,TCRR_TRAIN_WORKERS=0,
                         TCRR_VAL_WORKERS=2,TCRR_PERSISTENT_WORKERS=0,
                         TCRR_PREFETCH_FACTOR=1)
    assert cp.returncode == 0, cp.stderr
    assert len(calls)==2
    for call in calls:
        assert 'mem=32GB' in call
        v=params(call)
        assert v['BATCH_SIZE']=='32'
        assert v['TCRR_TRAIN_WORKERS']=='0'
        assert v['TCRR_VAL_WORKERS']=='2'
        assert v['TCRR_PERSISTENT_WORKERS']=='0'
        assert v['TCRR_PREFETCH_FACTOR']=='1'
    assert 'TRAIN_PBS_MEM=32GB' in cp.stdout
    assert 'POST_PBS_MEM=32GB' in cp.stdout


def test_smoke_full_batch_uses_default_64_without_manual_export(tmp_path):
    cp,calls=mock_submit(tmp_path,SMOKE=1,SMOKE_PROFILE_FULL_BATCH=1)
    assert cp.returncode == 0, cp.stderr
    assert len(calls)==1
    assert params(calls[0])['BATCH_SIZE']=='64'
    assert 'EFFECTIVE_SMOKE_BATCH_SIZE=64' in cp.stdout


def test_regular_smoke_remains_batch_three(tmp_path):
    cp,calls=mock_submit(tmp_path,SMOKE=1)
    assert cp.returncode == 0, cp.stderr
    assert len(calls)==1
    assert 'CONFIGURED_BATCH_SIZE=64' in cp.stdout
    assert 'EFFECTIVE_SMOKE_BATCH_SIZE=3' in cp.stdout


def test_python_yaml_defaults_match_launcher():
    import yaml
    cfg=yaml.safe_load((ROOT/'configs/tcrr_stla_v5.yaml').read_text())
    assert cfg['version']=='5.3.0'
    assert cfg['training']['batch_size']==64
    assert cfg['runtime']['num_workers']==4
    assert cfg['runtime']['val_num_workers']==0
    assert cfg['runtime']['persistent_workers'] is True
    assert cfg['runtime']['prefetch_factor']==2
    assert '#PBS -l mem=96GB' in (ROOT/'pbs/formal_v5.pbs').read_text()
    assert '#PBS -l mem=32GB' in (ROOT/'pbs/postprocess_v5.pbs').read_text()
