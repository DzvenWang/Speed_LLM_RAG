from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
cfg=yaml.safe_load((ROOT/'configs/baseline.yaml').read_text())
assert cfg['version']=='1.0.0'
assert cfg['protocol']['no_local_sampling'] is True
assert cfg['protocol']['shared_protocol_name']=='protocol_v1'
assert cfg['shared_contract']['required_version']=='1.0.0'
assert cfg['formal_selection_methods']==['ImputeFormer','LRTC-TMCP','GinAR','RAFT','RAST','SARAF','ReMFT']
print('PROTOCOL CONFIG SMOKE PASS')
