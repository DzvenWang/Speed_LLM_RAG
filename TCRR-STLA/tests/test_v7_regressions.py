"""Regressions for the actual failures in T7_speed.o182942163 and T7_flow.o182942169."""
from __future__ import annotations
import importlib.util,json,sys
from pathlib import Path
import numpy as np,pytest
PROJ=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJ))

def load_driver():
    name='tcrr_driver_for_test'
    spec=importlib.util.spec_from_file_location(name,PROJ/'scripts/run_v7.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_run_identity_tuple_roundtrip():
    d=load_driver()
    expected={'effective_runtime':{'retrieval_periods':(4,2,1),'workers':4},
              'configuration':{'training':{'amp':True}},'limits':{'train':None}}
    stored=d.canonical_json(expected)
    loaded=json.loads(json.dumps(stored))
    assert loaded==d.canonical_json(expected)
    assert loaded['effective_runtime']['retrieval_periods']==[4,2,1]
    assert not d.manifest_difference(loaded,stored)

def test_identity_mismatch_detected():
    d=load_driver()
    assert d.manifest_difference({'target':'speed'},{'target':'flow'})==['target']
    assert d.manifest_difference({'seed':10},{'seed':11})==['seed']

def test_teacher_diagnostic_only_existing_names():
    from tcrr_stla.v7.runner import summarize_train_response
    array={'normal':np.zeros((2,3,4),np.float32),
           'c2':np.ones((2,3,4),np.float32),
           'cf':np.full((2,3,4),2.,np.float32)}
    assert summarize_train_response(array)=={'normal':0.,'c2':1.,'cf':2.}
    with pytest.raises(RuntimeError,match='Teacher TRAIN cache missing keys'):
        summarize_train_response({'normal':array['normal']})

def test_all_regressions_have_isolated_version_namespace():
    source=(PROJ/'scripts/run_v7.py').read_text()
    assert 'RUN_NAMESPACE' in source
    assert 'V7_RUN_IDENTITY_PASS' in source
    assert 'VERSION' in source
    for p in ('pbs/formal_v7.pbs','pbs/submit_v7.sh'):
        assert 'v7_1_0_c1' in (PROJ/p).read_text()
