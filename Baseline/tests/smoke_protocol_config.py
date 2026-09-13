from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
cfg=yaml.safe_load((ROOT/'configs/baseline.yaml').read_text())
assert cfg['version']=='0.6.0'; assert cfg['protocol']['no_local_sampling'] is True; assert cfg['protocol']['shared_protocol_name']=='protocol_v1'; assert cfg['shared_contract']['required_version']=='1.0.0'
print('PROTOCOL CONFIG SMOKE PASS')
