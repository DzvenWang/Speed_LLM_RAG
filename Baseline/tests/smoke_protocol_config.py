from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from baseline.utils import load_yaml
cfg=load_yaml(ROOT/'configs/baseline.yaml')
assert cfg['protocol']['shared_protocol_name']=='protocol_core5_s40'
assert cfg['protocol']['shared_protocol_version']=='2.1.0'
assert cfg['protocol']['paired_group_id']=='TraffiDent_Net_06_core5_multiseverity_v1'
assert cfg['shared_contract']['required_version']=='1.2.0'
assert cfg['shared_contract']['required_cache_schema_version']==2
assert cfg['experiment']['families']==['entry_random','sensor_outage','temporal_outage','regional_outage','directional_outage']
assert cfg['experiment']['evaluation_severities']==[20,40,60,90]
print('SMOKE protocol config PASS')
