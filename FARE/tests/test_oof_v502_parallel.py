"""4-GPU scheduling, strict frozen-plan and checkpoint completion tests."""
import importlib.util
import json
import sys
from pathlib import Path
from unittest import mock
import numpy as np
import pandas as pd
import pytest
import torch
from fare.oof import save_plan, load_frozen_plan, validate_completed_fold,sha_json
from fare.config import hash_file


def load_submit_module():
    location=Path(__file__).parents[1]/'scripts'/'submit_oof.py'
    spec=importlib.util.spec_from_file_location('parallel_submit_oof',location)
    # _bootstrap lives in scripts folder
    sys.path.insert(0,str(location.parent))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parallel_pbs_job_graph(monkeypatch):
    sub=load_submit_module()
    received=[]
    def fake_qsub(cmd,text):
        received.append(cmd)
        return f'{100+len(received)}.gadi-pbs\n'
    monkeypatch.setattr(sub.subprocess,'check_output',fake_qsub)
    env={'TARGET':'speed','FARE_PYTHON':'/env/bin/python'}
    producers=[sub.submit('/fold.pbs',dict(env,OOF_FOLD_ID=str(i)),job_name=f'F{i}') for i in range(4)]
    finalize=sub.submit('/final.pbs',env,dependency=producers,job_name='FIN')
    fare=sub.submit('/fare.pbs',env,dependency=[finalize],job_name='FARE')
    assert len(received)==6
    assert all('-W' not in cmd for cmd in received[:4])
    assert received[4][received[4].index('-W')+1]=='depend=afterok:'+':'.join(producers)
    assert received[5][received[5].index('-W')+1]=='depend=afterok:'+finalize
    assert all('ngpus' not in ' '.join(cmd) for cmd in received)  # GPU in PBS, not qsub args


def test_frozen_plan_rejects_mismatched_folds_and_cache(tmp_path,monkeypatch):
    from fare.oof import row_keys
    from fare import oof
    from fare.protocol import SupportDataset
    # Use existing fixture constructing real cache from prior test
    from test_oof_v501 import create_cache
    cache,df=create_cache(tmp_path,nrows=100)
    out=tmp_path/'plan.json'
    p=save_plan(cache,None,'speed','TraffiDent_Net_06',out,nfolds=2,warmup=.5,purge=12)
    good=load_frozen_plan(out,cache,'speed','TraffiDent_Net_06',2,.5,12)
    assert good['plan_sha256']==p['plan_sha256']
    with pytest.raises(RuntimeError,match='flags/cache mismatch'):
        load_frozen_plan(out,cache,'speed','TraffiDent_Net_06',3,.5,12)
    tamper=json.loads(out.read_text());tamper['purge']=0;out.write_text(json.dumps(tamper))
    with pytest.raises(RuntimeError,match='Corrupted OOF plan'):
        load_frozen_plan(out,cache,'speed','TraffiDent_Net_06',2,.5,12)


def test_completed_fold_needs_verified_markers(tmp_path):
    from test_oof_v501 import create_cache
    cache,df=create_cache(tmp_path)
    plan=save_plan(cache,None,'speed','TraffiDent_Net_06',tmp_path/'fold_plan.json',nfolds=1,warmup=.5,purge=12)
    fold=plan['folds'][0]
    ref=tmp_path/'reference.pt';ref.write_bytes(b'reference')
    folder=tmp_path/'fold00';folder.mkdir()
    ck=folder/'best.pt';torch.save({'oof_provenance':{'plan_sha256':plan['plan_sha256'],'fold_id':0,
       'reference_checkpoint_sha256':hash_file(ref),'training_recipe_sha256':'abc',
       'fit_label_end':fold['fit_label_end'],'val_label_end':fold['val_label_end'],
       'evaluate_from':fold['evaluate_from'],'evaluate_until':fold['evaluate_until']}},ck)
    assert not validate_completed_fold(ck,plan,fold,ref,'abc')
    done={'plan_sha256':plan['plan_sha256'],'fold_id':0,
          'reference_checkpoint_sha256':hash_file(ref),'training_recipe_sha256':'abc',
          'checkpoint_sha256':hash_file(ck)}
    (folder/'COMPLETED.json').write_text(json.dumps(done))
    assert validate_completed_fold(ck,plan,fold,ref,'abc')
    (folder/'best.pt').write_bytes(b'tamper')
    with pytest.raises(Exception):validate_completed_fold(ck,plan,fold,ref,'abc')


def test_full_four_gpu_pipeline_submits_four_plus_finalize_plus_two_fare(tmp_path,monkeypatch):
    sub=load_submit_module()
    conf=tmp_path/'formal.yaml';conf.write_text('version: 5.0.3')
    monkeypatch.setattr(sub,'root',lambda:tmp_path)
    monkeypatch.setattr(sub,'project_path',lambda p,must_exist=False: Path(p) if Path(p).is_absolute() else tmp_path/p)
    monkeypatch.setattr(sub,'read_config',lambda path,dataset,target,seed:{
        'experiment':{'run_tag':'formal'},'evaluation':{'include_val':False}})
    monkeypatch.setattr(sub,'backbone_checkpoint',lambda cfg,override:tmp_path/'ref.pt')
    calls=[]
    monkeypatch.setattr(sub.subprocess,'check_call',lambda cmd:calls.append(('plan',cmd)))
    def qsub(cmd,text):
        calls.append(('qsub',cmd))
        return f'{1800000+len([c for c in calls if c[0]=="qsub"])}.gadi-pbs\n'
    monkeypatch.setattr(sub.subprocess,'check_output',qsub)
    monkeypatch.setattr(sys,'argv',['submit_oof.py','--target','both','--nfolds','2',
                                   '--warmup','0.30','--epochs','100','--parallel-folds','--auto-fare'])
    sub.main()
    assert calls[0][0]=='plan'
    all_jobs=[x[1] for x in calls if x[0]=='qsub']
    assert len(all_jobs)==7
    for ix in range(4):
        assert any('OOF_FOLD_ID='+str(ix%2) in v for v in all_jobs[ix] if isinstance(v,str))
        assert 'oof_fold.pbs' in all_jobs[ix][-1]
    assert 'oof_finalize.pbs' in all_jobs[4][-1]
    assert all('full_pipeline.pbs' in all_jobs[i][-1] for i in (5,6))
    assert all_jobs[4][all_jobs[4].index('-W')+1].count('gadi-pbs')==4
    assert all_jobs[5][all_jobs[5].index('-W')+1]=='depend=afterok:1800005.gadi-pbs'
    assert all_jobs[6][all_jobs[6].index('-W')+1]=='depend=afterok:1800005.gadi-pbs'
    receipts=list((tmp_path/'FARE_exps/oof_submissions').glob('*.json'))
    assert len(receipts)==1
    receipt=json.loads(receipts[0].read_text())
    assert receipt['status']=='submitted'
    assert len(receipt['jobs']['producers'])==4
    assert len(receipt['jobs']['fare'])==2
