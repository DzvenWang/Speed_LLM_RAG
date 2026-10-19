from pathlib import Path
from fare.config import read_config

ROOT=Path(__file__).resolve().parents[1]

def test_formal_defaults():
    cfg=read_config(str(ROOT/'configs/formal.yaml'))
    assert cfg['version']=='5.0.3'
    assert cfg['memory']['residual_mode']=='external_oos_paired'
    assert cfg['memory']['require_oos_for_formal']
    assert cfg['evaluation']['auto_evaluate']
    assert set(cfg['experiment']['severities'])=={20,40,60,90}


def test_pbs_resources_and_formal_config():
    p=(ROOT/'pbs/full_pipeline.pbs').read_text()
    assert 'gpuvolta' in p and 'ncpus=12' in p and 'ngpus=1' in p
    assert 'FARE/configs/formal.yaml' in p
    assert 'FARE_PYTHON' in p


def test_no_v4_teacher_import_in_training():
    s=(ROOT/'fare/training.py').read_text()
    assert 'utility_teacher' in s
    assert 'generate_utility_teacher' in s
    assert 'replay_utility_loss' in s
