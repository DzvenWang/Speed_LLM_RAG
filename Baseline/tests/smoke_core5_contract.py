#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from baseline.utils import load_yaml
cfg=load_yaml(ROOT/'configs/baseline.yaml')
expected=['entry_random','sensor_outage','temporal_outage','regional_outage','directional_outage']
assert cfg['version']=='1.1.0'
assert cfg['experiment']['severity']==0.4
assert cfg['experiment']['evaluation_severities']==[20,40,60,90]
assert cfg['experiment']['families']==expected
assert cfg['protocol']['shared_protocol_name']=='protocol_core5_s40'
assert cfg['protocol']['shared_protocol_version']=='2.1.0'
assert cfg['shared_contract']['required_version']=='1.2.0'
assert cfg['shared_contract']['required_cache_schema_version']==2
assert cfg['frozen_backbone_contract']['required_project_version']=='1.0.10'
print('CORE5 BASELINE CONTRACT SMOKE PASS')
