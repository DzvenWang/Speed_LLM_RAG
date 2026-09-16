#!/usr/bin/env python3
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
cfg=yaml.safe_load((ROOT/'configs/baseline.yaml').read_text())
primary=['ImputeFormer','LRTC-TMCP','GinAR','RAFT','RAST','SARAF','ReMFT']
secondary=['SPRINT','SLPF','MoGERNN','LEAF','LEAF-Qwen1.5B','RAST-PatchSTG']
assert cfg['version']=='1.0.0'
assert cfg['formal_selection_methods']==primary
assert cfg['baseline_story']['main_table']==primary
assert cfg['secondary_methods']==secondary
assert 'compatible_result_project_versions' not in cfg
assert 'implementation_revision' not in cfg
assert 'method_implementation_revisions' not in cfg
assert 'development_cache_migration_from' not in cfg['protocol']
for m in primary:
    assert cfg['methods'][m]['primary_experiment'] is True
    assert cfg['methods'][m]['table_group']=='main'
for m in secondary:
    assert cfg['methods'][m]['primary_experiment'] is False
    assert cfg['methods'][m]['table_group']=='secondary'
# GinAR invalid formal runs must fail before artifact selection.
g=(ROOT/'baseline/runners/ginar_runner.py').read_text()
assert '--stability-probe' in g
assert "raise RuntimeError(msg + ' -- run rejected')" in g
assert float(cfg['runners']['GinAR']['formal_learning_rate'])==5e-4
assert float(cfg['runners']['GinAR']['gradient_clip'])==1.0
# ReMFT must remove random frozen layers and then require exact checkpoint/runtime depth.
r=(ROOT/'baseline/runners/remft_runner.py').read_text()
assert 'def enforce_checkpoint_consistent_bert_depth' in r
assert 'trimmed_random_runtime_layers' in r
assert 'runtime_matches_local_depth' in r
assert 'depth mismatch remains after alignment' in r
print('CLEAN PRIMARY CONTRACT PASS')
