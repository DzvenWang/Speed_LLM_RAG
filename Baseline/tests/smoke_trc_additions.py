#!/usr/bin/env python3
from pathlib import Path
import sys,yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baseline.registry import METHODS
cfg=yaml.safe_load((ROOT/'configs/baseline.yaml').read_text())
# Recent TRC representatives remain implemented, but only LRTC-TMCP and ReMFT are primary.
for m,f in [('LRTC-TMCP','lrtc_tmcp_runner.py'),('SPRINT','sprint_runner.py'),('MoGERNN','mogernn_runner.py'),('ReMFT','remft_runner.py')]:
    assert m in METHODS
    assert (ROOT/'baseline/runners'/f).exists()
assert cfg['methods']['LRTC-TMCP']['table_group']=='main'
assert cfg['methods']['ReMFT']['table_group']=='main'
assert cfg['methods']['SPRINT']['table_group']=='secondary'
assert cfg['methods']['MoGERNN']['table_group']=='secondary'
print('TRC ADDITIONS TAXONOMY PASS')
