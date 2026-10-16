"""Regression tests for FARE v3.0.2 scheduling + post-training scopes."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def test_post_training_defaults_to_four_tests_only(monkeypatch):
    import fare.pipeline as pipeline
    monkeypatch.setattr(pipeline,'evaluations',lambda c,limit_override=None: {
        's20':'test20','s40':'test40','s60':'test60','s90':'test90'
    })
    cfg={'evaluation':{'include_val':False}}
    assert list(pipeline.post_training_datasets(cfg,'validation'))==['s20','s40','s60','s90']
    cfg['evaluation']['include_val']=True
    assert list(pipeline.post_training_datasets(cfg,'validation'))==['val','s20','s40','s60','s90']


def test_default_pilot_config_disables_post_training_val():
    from fare.config import read_config
    c=read_config(Path(__file__).resolve().parents[1]/'configs'/'pilot.yaml')
    assert c['evaluation']['auto_evaluate'] is True
    assert c['evaluation']['include_val'] is False
    assert c['experiment']['severities']==[20,40,60,90]
    assert c['analysis']['counterfactual_split']=='s40'


def test_submit_gadi_calls_qsub_exactly_once(tmp_path,monkeypatch,capsys):
    """Only full_pipeline.pbs is submitted; never an analysis afterok job."""
    import fare.config as config
    from pathlib import Path
    (tmp_path/'FARE/pbs').mkdir(parents=True,exist_ok=True)
    (tmp_path/'FARE/pbs/full_pipeline.pbs').write_text('#!/bin/bash\n')
    monkeypatch.setenv('SPEED_LLM_RAG_ROOT',str(tmp_path))
    script=Path(__file__).resolve().parents[1]/'scripts'/'submit_gadi.py'
    assert script.is_file()
    # Script normally imports local bootstrap from its own script directory.
    monkeypatch.syspath_prepend(str(script.parent))
    spec=importlib.util.spec_from_file_location('_fare_v302_submit',script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls=[]
    def mock_check_output(cmd, text=True):
        calls.append(cmd)
        return '199999999.gadi-pbs\n'
    monkeypatch.setattr(module.subprocess,'check_output',mock_check_output)
    source=Path(__file__).resolve().parents[1]/'configs'/'pilot.yaml'
    monkeypatch.setattr(sys,'argv',['submit_gadi.py','--config',str(source),'--target','flow','--seed','10','--skip-precheck'])
    module.main()
    assert len(calls)==1
    assert calls[0][0]=='qsub'
    assert calls[0][-1].endswith('FARE/pbs/full_pipeline.pbs')
    assert not any('depend=afterok' in str(x) for x in calls[0])
    variables=calls[0][calls[0].index('-v')+1]
    assert 'FARE_EVAL_VAL=0' in variables
    assert 'FARE_PYTHON=' in variables
    assert 'FARE_ENV_PREFIX=' in variables
    output=capsys.readouterr().out
    assert 'FARE4_FULL_PIPELINE_JOB=199999999.gadi-pbs' in output
    assert 'ANALYSIS_JOB' not in output
    assert 'FARE4_POST_TRAINING_EVALUATION=test_only' in output


def test_submit_gadi_eval_val_override(tmp_path,monkeypatch):
    """Optional complete eval still one job."""
    import fare.config as config
    script=Path(__file__).resolve().parents[1]/'scripts'/'submit_gadi.py'
    (tmp_path/'FARE/pbs').mkdir(parents=True,exist_ok=True)
    (tmp_path/'FARE/pbs/full_pipeline.pbs').write_text('#!/bin/bash\n')
    monkeypatch.setenv('SPEED_LLM_RAG_ROOT',str(tmp_path))
    monkeypatch.syspath_prepend(str(script.parent))
    spec=importlib.util.spec_from_file_location('_fare_v302_submit_val',script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls=[]
    monkeypatch.setattr(module.subprocess,'check_output',lambda cmd,text=True:(calls.append(cmd) or '200000000.gadi-pbs\n'))
    source=Path(__file__).resolve().parents[1]/'configs'/'pilot.yaml'
    monkeypatch.setattr(sys,'argv',['submit_gadi.py','--config',str(source),'--target','speed','--eval-val','--skip-precheck'])
    module.main()
    assert len(calls)==1
    assert 'FARE_EVAL_VAL=1' in calls[0][2]


def test_submit_default_precheck_blocks_qsub_if_invalid(tmp_path, monkeypatch):
    """A failing CPU safety gate must prevent wasting a queued V100 job."""
    import subprocess
    script=Path(__file__).resolve().parents[1]/'scripts'/'submit_gadi.py'
    monkeypatch.syspath_prepend(str(script.parent))
    spec=importlib.util.spec_from_file_location('_fare_gate_blocks',script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source=Path(__file__).resolve().parents[1]/'configs'/'pilot.yaml'
    calls=[]
    def failed_check(cmd):
        calls.append(('gate',cmd))
        raise subprocess.CalledProcessError(9,cmd)
    monkeypatch.setattr(module.subprocess,'check_call',failed_check)
    monkeypatch.setattr(module.subprocess,'check_output',lambda *a,**kw: calls.append(('qsub',a)))
    monkeypatch.setattr(sys,'argv',['submit_gadi.py','--config',str(source),
                                      '--target','speed'])
    import pytest
    with pytest.raises(subprocess.CalledProcessError):
        module.main()
    assert [x[0] for x in calls]==['gate']


def test_submit_check_only_never_qsub(tmp_path,monkeypatch,capsys):
    """User can verify the environment without requesting any cluster resources."""
    script=Path(__file__).resolve().parents[1]/'scripts'/'submit_gadi.py'
    monkeypatch.syspath_prepend(str(script.parent))
    spec=importlib.util.spec_from_file_location('_fare_check_only',script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source=Path(__file__).resolve().parents[1]/'configs'/'pilot.yaml'
    calls=[]
    monkeypatch.setattr(module.subprocess,'check_call',lambda cmd:calls.append(('check',cmd)))
    monkeypatch.setattr(module.subprocess,'check_output',lambda *args,**kwargs:calls.append(('qsub',args)))
    monkeypatch.setattr(sys,'argv',['submit_gadi.py','--config',str(source),
                                      '--target','speed','--check-only'])
    module.main()
    assert len(calls)==1 and calls[0][0]=='check'
    assert 'FARE_CHECK_ONLY_NO_QSUB' in capsys.readouterr().out
