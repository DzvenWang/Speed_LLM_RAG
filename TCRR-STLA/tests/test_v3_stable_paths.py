"""Release regression for stable project directories and cache isolation."""
from pathlib import Path
import os

def test_stable_project_roots(monkeypatch, tmp_path):
    monkeypatch.setenv('SPEED_LLM_RAG_ROOT', str(tmp_path))
    monkeypatch.setenv('TRAFFIDENT_TARGET', 'speed')
    monkeypatch.setenv('BACKBONE_MODE', 'incomplete')
    from tcrr_stla.config import load_config
    cfg = load_config()
    assert Path(cfg['project']['exp_root']).parts[-3:] == ('TCRR-STLA_exps', 'speed', 'incomplete')
    assert Path(cfg['project']['log_root']).parts[-3:] == ('TCRR-STLA_logs', 'speed', 'incomplete')
    assert Path(cfg['project']['retrieval_root']).parts[-3:] == ('TCRR-STLA_cache', 'speed', 'incomplete')
    assert '_v3_' not in cfg['project']['retrieval_root']

def test_replay_cache_formal_smoke_isolation():
    from tcrr_stla.replay_cache import replay_bank_dir
    cfg={'project':{'retrieval_root':'/tmp/TCRR-STLA_cache/speed/incomplete'},
         'experiment':{'dataset':'TraffiDent_Net_06'}}
    formal=replay_bank_dir(cfg,False)
    smoke=replay_bank_dir(cfg,True)
    assert formal!=smoke
    assert formal.parts[-3:]==('v3','formal','teacher_replay_v2')
    assert smoke.parts[-3:]==('v3','formal_smoke','teacher_replay_v2')

def test_retrieval_root_is_v3_separated(tmp_path):
    import inspect
    from scripts.build_retrieval_cache import ensure_cache
    assert "'v3' if str(cfg.get('version',''))" in inspect.getsource(ensure_cache)
