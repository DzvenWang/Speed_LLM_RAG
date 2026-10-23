"""Enforce: reuse OOF + STLAformer only, never warm-start old FARE weights."""
from pathlib import Path
import subprocess,sys
import pytest,yaml
from fare.config import read_config
from fare.training import fit

ROOT=Path(__file__).resolve().parents[1]

def test_fresh_config_default():
    cfg=read_config(ROOT/'configs/formal.yaml')
    assert not cfg['training'].get('initialize_from')
    assert cfg['memory']['residual_mode']=='external_oos_paired'

def test_config_rejects_warm_start(tmp_path):
    cfg=yaml.safe_load((ROOT/'configs/formal.yaml').read_text())
    cfg['training']['initialize_from']='/tmp/previous_FARe_best.pt'
    p=tmp_path/'config.yaml';p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='fresh-init'):
        read_config(p)

def test_submit_rejects_init_from():
    p=subprocess.run([sys.executable,str(ROOT/'scripts/submit_gadi.py'),
                     '--target','speed','--init-from','prior_best.pt','--check-only'],
                     capture_output=True,text=True)
    assert p.returncode!=0
    assert 'unrecognized arguments' in p.stderr

def test_training_runner_rejects_init_from():
    p=subprocess.run([sys.executable,str(ROOT/'scripts/run_pipeline.py'),
                     '--target','speed','--init-from','prior_best.pt'],
                     capture_output=True,text=True)
    assert p.returncode!=0
    assert 'unrecognized arguments' in p.stderr

def test_pbs_rejects_inherited_warm_start():
    s=(ROOT/'pbs/full_pipeline.pbs').read_text()
    assert 'FARE_INIT_FROM' in s
    assert 'warm-start from older FARE weights is forbidden' in s
    assert 'CMD+=(--init-from' not in s

def test_fresh_run_directory_collision_guard():
    s=(ROOT/'fare/pipeline.py').read_text()
    assert "('best.pt', 'last.pt', 'training_complete.json')" in s
    assert 'FileExistsError' in s
