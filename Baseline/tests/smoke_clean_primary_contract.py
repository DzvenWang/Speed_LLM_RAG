#!/usr/bin/env python3
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
cfg=yaml.safe_load((ROOT/'configs/baseline.yaml').read_text())
primary=['ImputeFormer','LRTC-TMCP','GinAR','RAFT','RAST','SARAF','ReMFT']
secondary=['SPRINT','SLPF','MoGERNN','LEAF','LEAF-Qwen1.5B','RAST-PatchSTG','RAFT-PatchSTG','SARAF-PatchSTG']
assert cfg['version']=='1.1.0'
assert cfg['formal_selection_methods']==primary
assert cfg['baseline_story']['main_table']==primary
assert cfg['secondary_methods']==secondary
assert 'compatible_result_project_versions' not in cfg
assert 'implementation_revision' not in cfg
assert 'ReMFT' in cfg.get('method_implementation_revisions',{})
assert 'development_cache_migration_from' not in cfg['protocol']
for m in primary:
    assert cfg['methods'][m]['primary_experiment'] is True
    assert cfg['methods'][m]['table_group']=='main'
for m in secondary:
    assert cfg['methods'][m]['primary_experiment'] is False
    assert cfg['methods'][m]['table_group']=='secondary'
# GinAR stability probes remain fail-fast, while Formal post-best divergence
# is a guarded early-stop that preserves the previously selected Val-best checkpoint.
g=(ROOT/'baseline/runners/ginar_runner.py').read_text()
assert '--stability-probe' in g
assert "raise RuntimeError(msg + ' -- stability probe rejected')" in g
assert 'GINAR DIVERGENCE GUARD EARLY-STOP' in g
assert "'selection_valid': True" in g
assert float(cfg['runners']['GinAR']['formal_learning_rate'])==5e-4
assert float(cfg['runners']['GinAR']['gradient_clip'])==1.0
# ReMFT must remove random frozen layers and then require exact checkpoint/runtime depth.
r=(ROOT/'baseline/runners/remft_runner.py').read_text()
assert 'def normalize_bert_depth_to_checkpoint' in r
assert 'trimmed_random_layers' in r
assert 'runtime_matches_local_depth' in r
assert 'runtime/checkpoint BERT depth mismatch after normalization' in r
print('CLEAN PRIMARY CONTRACT PASS')
