"""PBS parameter forwarding and isolated full-batch smoke regression tests."""
from __future__ import annotations
import json,os,subprocess,sys
from pathlib import Path
import pytest
import torch
from torch.utils.data import TensorDataset
from tcrr_stla.v5.shared_adapter import (
    effective_training_batch_size,smoke_training_dataset,s0_identity)

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent
SCRIPT=ROOT/'pbs/submit_pipeline_v5.sh'


def submit_mock(tmp_path,**overrides):
    fake=tmp_path/'bin';fake.mkdir()
    qsub=fake/'qsub'
    qsub.write_text('#!/usr/bin/env python3\n'
                    'import json,os,sys\n'
                    'with open(os.environ["QSUB_CALLS"],"a") as f: '\
                    'f.write(json.dumps(sys.argv[1:])+"\\n")\n'
                    'print("182555001.gadi-pbs")\n')
    qsub.chmod(0o755)
    ck=tmp_path/'source_best.pt';ck.write_bytes(b'checkpoint stub')
    log=tmp_path/'calls.jsonl'
    env=os.environ.copy()
    for key in ('BATCH_SIZE','TCRR_TRAIN_WORKERS','TCRR_EVAL_WORKERS',
                'DIAGNOSTIC_BATCH_SIZE','DIAGNOSTIC_MAX_BATCHES',
                'DIAGNOSTIC_PRIMARY_MAX_BATCHES','SMOKE','SMOKE_PROFILE_FULL_BATCH'):
        env.pop(key,None)
    env.update(PATH=str(fake)+os.pathsep+env.get('PATH',''),QSUB_CALLS=str(log),
               SPEED_LLM_RAG_ROOT=str(REPO),RUN_STAMP='20991231_230501',
               TARGET='speed',BACKBONE_CHECKPOINT=str(ck))
    env.update({key:str(val) for key,val in overrides.items()})
    result=subprocess.run(['bash',str(SCRIPT)],cwd=REPO,env=env,
                          capture_output=True,text=True)
    calls=[json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result,calls


def _vars(call):
    return dict(item.split('=',1) for item in call[call.index('-v')+1].split(','))


def test_full_pipeline_passes_all_runtime_settings(tmp_path):
    res,calls=submit_mock(tmp_path,BATCH_SIZE=64,TCRR_TRAIN_WORKERS=2,
                          TCRR_EVAL_WORKERS=1,DIAGNOSTIC_BATCH_SIZE=12,
                          DIAGNOSTIC_MAX_BATCHES=0,DIAGNOSTIC_PRIMARY_MAX_BATCHES=6)
    assert res.returncode==0,res.stderr
    assert len(calls)==2
    for call in calls:
        settings=_vars(call)
        assert settings['BATCH_SIZE']=='64'
        assert settings['TCRR_TRAIN_WORKERS']=='2'
        assert settings['TCRR_EVAL_WORKERS']=='1'
        assert settings['DIAGNOSTIC_BATCH_SIZE']=='12'
        assert settings['DIAGNOSTIC_MAX_BATCHES']=='0'
        assert settings['DIAGNOSTIC_PRIMARY_MAX_BATCHES']=='6'
        assert settings['TARGET']=='speed'
        assert settings['SEED']=='10'
    assert 'SKIP_EVALUATE=1' in calls[0][calls[0].index('-v')+1]
    assert 'depend=afterok:182555001.gadi-pbs' in calls[1]
    assert 'TCRR-STLA/pbs/postprocess_v5.pbs' in calls[1]
    assert 'BATCH_SIZE=64' in res.stdout


def test_full_batch_smoke_submits_single_job_and_isolates_outputs(tmp_path):
    res,calls=submit_mock(tmp_path,SMOKE=1,SMOKE_PROFILE_FULL_BATCH=1,BATCH_SIZE=64)
    assert res.returncode==0,res.stderr
    assert len(calls)==1
    assert '-l' in calls[0] and 'walltime=04:00:00' in calls[0]
    opts=_vars(calls[0]);assert opts['SMOKE']=='1'
    assert opts['SMOKE_PROFILE_FULL_BATCH']=='1'
    assert opts['BATCH_SIZE']=='64'
    assert '/v5/smoke/' in opts['RUN_DIR']
    assert 'SMOKE_ID=' in res.stdout


@pytest.mark.parametrize('kwargs,expected',[
    (dict(BATCH_SIZE=0),'BATCH_SIZE'),
    (dict(BATCH_SIZE=-1),'BATCH_SIZE'),
    (dict(BATCH_SIZE='64,SKIP_DIAGNOSTICS=0'),'BATCH_SIZE'),
    (dict(TCRR_TRAIN_WORKERS=9),'TCRR_TRAIN_WORKERS'),
    (dict(DIAGNOSTIC_BATCH_SIZE=0),'DIAGNOSTIC_BATCH_SIZE'),
    (dict(SMOKE_PROFILE_FULL_BATCH=1,BATCH_SIZE=64),'requires SMOKE=1'),
])
def test_bad_launch_arguments_fail_before_submitting(tmp_path,kwargs,expected):
    res,calls=submit_mock(tmp_path,**kwargs)
    assert res.returncode!=0
    assert not calls
    assert expected in res.stderr


def test_training_batch_size_smoke_modes_are_explicit():
    cfg={'training':{'batch_size':64},'runtime':{}}
    train=TensorDataset(torch.arange(24))
    assert effective_training_batch_size(cfg,smoke=True)==3
    assert len(smoke_training_dataset(train,cfg,smoke=True))==24
    cfg['runtime']['smoke_full_batch']=True
    assert effective_training_batch_size(cfg,smoke=True)==64
    samples=smoke_training_dataset(train,cfg,smoke=True)
    assert len(samples)==64
    assert samples[0][0].item()==0 and samples[24][0].item()==0
    assert s0_identity(cfg,smoke=True)['training_batch_size']==64
    assert effective_training_batch_size(cfg,smoke=False)==64
    assert len(smoke_training_dataset(train,cfg,smoke=False))==24


def test_launcher_to_python_contract_is_concrete():
    formal=(ROOT/'pbs/formal_v5.pbs').read_text()
    training=(ROOT/'scripts/train_v5.py').read_text()
    trainer=(ROOT/'tcrr_stla/v5/trainer.py').read_text()
    shared=(ROOT/'tcrr_stla/v5/shared_adapter.py').read_text()
    assert 'ARGS+=(--batch-size "$BATCH_SIZE")' in formal
    assert 'ARGS+=(--smoke-full-batch)' in formal
    assert "p.add_argument('--smoke-full-batch'" in training
    assert "cfg['runtime']['smoke_full_batch']=bool(a.smoke_full_batch)" in training
    assert 'smoke_training_dataset(train_ds,cfg,smoke)' in trainer
    assert 'smoke_training_dataset(train_ds,cfg,smoke)' in shared
