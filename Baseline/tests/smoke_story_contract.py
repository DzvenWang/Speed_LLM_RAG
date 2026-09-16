#!/usr/bin/env python3
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
cfg=yaml.safe_load((ROOT/'configs/baseline.yaml').read_text())
main=['ImputeFormer','LRTC-TMCP','GinAR','RAFT','RAST','SARAF','ReMFT']
assert cfg['baseline_story']['main_table']==main
assert cfg['formal_selection_methods']==main
for m in main:
    assert cfg['methods'][m]['table_group']=='main'
for m in ['SPRINT','SLPF','MoGERNN','LEAF']:
    assert cfg['methods'][m]['table_group']=='secondary'
print('STORY CONTRACT PASS')
