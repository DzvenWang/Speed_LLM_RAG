"""Real qsub invocation mock for CPU/memory and DataLoader tuning passthrough."""
import json,os,subprocess
from pathlib import Path
import pytest
from torch.utils.data import TensorDataset
import torch
from tcrr_stla.trainer import _make_loader

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent

def run_mock(tmp_path, **params):
    fake=tmp_path/'bin';fake.mkdir()
    q=fake/'qsub'; q.write_text('#!/usr/bin/env python3\nimport os,sys,json\n'
       'with open(os.environ["QSUB_LOG"], "a") as o:o.write(json.dumps(sys.argv[1:])+"\\n")\n'
       'print("182600000.gadi-pbs")\n')
    q.chmod(0o755)
    ck=tmp_path/'best.pt'; ck.write_bytes(b'ck')
    log=tmp_path/'calls.txt'
    env=os.environ.copy()
    for k in ('BATCH_SIZE','SMOKE','SMOKE_PROFILE_FULL_BATCH','TCRR_PBS_MEM','TCRR_PBS_NCPUS',
              'TCRR_TRAIN_WORKERS','TCRR_VAL_WORKERS','TCRR_PERSISTENT_WORKERS','TCRR_PREFETCH_FACTOR'):
        env.pop(k,None)
    env.update(PATH=str(fake)+os.pathsep+env['PATH'],QSUB_LOG=str(log),SPEED_LLM_RAG_ROOT=str(REPO),
               RUN_STAMP='20991231_222222',TARGET='speed',BACKBONE_CHECKPOINT=str(ck))
    env.update({k:str(v) for k,v in params.items()})
    result=subprocess.run(['bash',str(ROOT/'pbs/submit_pipeline_v5.sh')], cwd=REPO,
                          env=env, capture_output=True,text=True)
    calls=[json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
    return result,calls

def vals(call):
    return dict(kv.split('=',1) for kv in call[call.index('-v')+1].split(','))

def test_96gb_and_worker_tuning_passes_to_all_jobs(tmp_path):
    res,calls=run_mock(tmp_path,BATCH_SIZE=64,TCRR_PBS_MEM='96GB',TCRR_PBS_NCPUS=12,
                       TCRR_TRAIN_WORKERS=4,TCRR_VAL_WORKERS=0,
                       TCRR_PERSISTENT_WORKERS=1,TCRR_PREFETCH_FACTOR=2)
    assert res.returncode==0,res.stderr
    assert len(calls)==2
    for c in calls:
        assert 'ncpus=12' in c
        v=vals(c)
        for k,expected in [('BATCH_SIZE','64'),('TCRR_TRAIN_WORKERS','4'),('TCRR_VAL_WORKERS','0'),
                           ('TCRR_PERSISTENT_WORKERS','1'),('TCRR_PREFETCH_FACTOR','2')]:
            assert v[k]==expected
    assert 'mem=96GB' in calls[0]
    assert 'mem=32GB' in calls[1]
    assert 'TRAIN_PBS_MEM=96GB' in res.stdout
    assert 'POST_PBS_MEM=32GB' in res.stdout

def test_memory_applies_to_smoke_only(tmp_path):
    res,calls=run_mock(tmp_path,SMOKE=1,BATCH_SIZE=64,SMOKE_PROFILE_FULL_BATCH=1,TCRR_PBS_MEM='96GB')
    assert res.returncode==0,res.stderr
    assert len(calls)==1
    assert 'mem=96GB' in calls[0]

@pytest.mark.parametrize('params',[
    {'TCRR_PBS_MEM':'bad'}, {'TCRR_PBS_MEM':'512GB'},
    {'TCRR_PBS_NCPUS':13}, {'TCRR_TRAIN_WORKERS':10},
    {'TCRR_VAL_WORKERS':9}, {'TCRR_PERSISTENT_WORKERS':2},
    {'TCRR_PREFETCH_FACTOR':0},
])
def test_invalid_tuning_rejected(tmp_path,params):
    res,calls=run_mock(tmp_path,**params)
    assert res.returncode!=0
    assert calls==[]

def test_dataloader_persistent_and_prefetch_works():
    ds=TensorDataset(torch.arange(18))
    cfg={'runtime':{'num_workers':2,'persistent_workers':True,'prefetch_factor':2,'pin_memory':False}}
    loader=_make_loader(ds,4,False,cfg)
    assert loader.num_workers==2
    assert loader.persistent_workers is True
    assert loader.prefetch_factor==2
    assert sum(batch[0].numel() for batch in loader)==18
    it=getattr(loader,'_iterator',None)
    if it is not None:it._shutdown_workers()

def test_training_val_worker_pool_separation_in_source():
    src=(ROOT/'tcrr_stla/v5/trainer.py').read_text()
    assert "'val_num_workers'" in src
    assert 'False,val_cfg,smoke' in src
