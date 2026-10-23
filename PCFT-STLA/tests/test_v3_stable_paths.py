"""Release regression for stable project directories and cache isolation."""
from pathlib import Path
import os

def test_stable_project_roots(monkeypatch, tmp_path):
    monkeypatch.setenv('SPEED_LLM_RAG_ROOT', str(tmp_path))
    monkeypatch.setenv('TRAFFIDENT_TARGET', 'speed')
    monkeypatch.setenv('BACKBONE_MODE', 'incomplete')
    from pcft_stla.config import load_config
    cfg = load_config()
    assert Path(cfg['project']['exp_root']).parts[-3:] == ('PCFT-STLA_exps', 'speed', 'incomplete')
    assert Path(cfg['project']['log_root']).parts[-3:] == ('PCFT-STLA_logs', 'speed', 'incomplete')
    assert Path(cfg['project']['store_root']).parts[-3:] == ('PCFT-STLA_cache', 'speed', 'incomplete')
    assert '_v3_' not in cfg['project']['store_root']

def test_pcft_graph_version_isolation():
    from pcft_stla.trainer import dataset_store_dir
    cfg={'project':{'store_root':'/tmp/PCFT-STLA_cache/speed/incomplete'},
         'experiment':{'dataset':'TraffiDent_Net_06'}}
    assert dataset_store_dir(cfg).parts[-2:]==('TraffiDent_Net_06','v3')
