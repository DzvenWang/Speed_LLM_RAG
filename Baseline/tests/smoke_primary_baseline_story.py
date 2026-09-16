#!/usr/bin/env python3
from pathlib import Path
import sys, yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baseline.registry import MAIN_ORDER, SECONDARY_ORDER, METHODS
cfg=yaml.safe_load((ROOT/'configs/baseline.yaml').read_text())
expected=['ImputeFormer','LRTC-TMCP','GinAR','RAFT','RAST','SARAF','ReMFT']
assert cfg['baseline_story']['main_table']==expected, cfg['baseline_story']['main_table']
assert cfg['formal_selection_methods']==expected, cfg['formal_selection_methods']
assert MAIN_ORDER==expected, MAIN_ORDER
for m in expected:
    assert cfg['methods'][m]['enabled'] is True
    assert cfg['methods'][m]['table_group']=='main'
    assert cfg['methods'][m]['primary_experiment'] is True
    assert METHODS[m]['table_group']=='main'
secondary=['SPRINT','SLPF','MoGERNN','LEAF']
for m in secondary:
    assert m in SECONDARY_ORDER
    assert cfg['methods'][m]['table_group']=='secondary'
    assert cfg['methods'][m]['primary_experiment'] is False
print('PRIMARY BASELINE STORY SMOKE PASS')
