from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
cfg=yaml.safe_load((ROOT/'configs/baseline.yaml').read_text())
assert cfg['version']=='0.6.0'
assert cfg['shared_contract']['required_version']=='1.0.0'
assert cfg['shared_contract']['data_contract_revision']=='traffident_experiment_cache_v1'
assert cfg['frozen_backbone_contract']['required_project_version']=='0.5.0'
assert cfg['frozen_backbone_contract']['use_reference_checkpoint_for_all_baseline_seeds'] is True
print('V060 FORMAL CONTRACT SMOKE PASS')
