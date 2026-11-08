"""The v5 project must never route new outputs into historical TCRR folders."""
from pathlib import Path
import os
import re
import yaml
from tcrr_stla.config import load_config
from tcrr_stla.replay_cache import replay_bank_dir
from tcrr_stla.v5.shared_adapter import s0_settings

ROOT=Path(__file__).resolve().parents[1]


def test_canonical_version_and_config_key(monkeypatch,tmp_path):
    monkeypatch.setenv('SPEED_LLM_RAG_ROOT',str(tmp_path))
    monkeypatch.setenv('TRAFFIDENT_TARGET','speed')
    monkeypatch.setenv('BACKBONE_MODE','incomplete')
    cfg=load_config()
    assert cfg['version']=='5.3.0'
    assert 'v5' in cfg and 'v41' not in cfg
    assert (ROOT/'VERSION').read_text().strip()=='5.3.0'
    assert Path(cfg['project']['exp_root']).parts[-3:]==('TCRR-STLA_exps','speed','incomplete')
    path=replay_bank_dir(cfg,False)
    assert path.parts[-3:]==('v5','formal','teacher_replay')
    assert 'v41' not in str(path)


def test_active_launchers_and_checkpoints_are_version_strict():
    for filename in ('scripts/train_v5.py','scripts/evaluate_v5.py',
                     'scripts/auto_diagnose_v5.py','pbs/formal_v5.pbs',
                     'configs/tcrr_stla_v5.yaml'):
        assert (ROOT/filename).is_file()
    trainer=(ROOT/'scripts/train_v5.py').read_text()
    evaluation=(ROOT/'scripts/evaluate_v5.py').read_text()
    assert "cfg['version']!='5.3.0'" in trainer
    assert "ck.get('version')!='5.3.0'" in evaluation
    assert "train_v5.py" in (ROOT/'pbs/formal_v5.pbs').read_text()
    assert 'auto_diagnose_v5.py' in trainer
    assert 'tcrr_stla.v4' not in trainer
    assert 'tcrr_stla.v4' not in evaluation
    assert 'from ..v4' not in (ROOT/'tcrr_stla/v5/trainer.py').read_text()


def test_v5_cache_and_s0_contract():
    source=(ROOT/'tcrr_stla/v5/shared_adapter.py').read_text()
    assert "payload.get('version')!=cfg['version']" in source
    assert s0_settings({'v5':{'s0_lr':0.0004,'s0_epochs':7}})==(0.0004,7)
    assert "('v5' if" in (ROOT/'tcrr_stla/normal_teacher.py').read_text()
    assert "('v5' if" in (ROOT/'tcrr_stla/replay_cache.py').read_text()
    assert 'folder, prefix = \'v5\', \'tcrr_v5_\'' in (ROOT/'scripts/build_retrieval_cache.py').read_text()
