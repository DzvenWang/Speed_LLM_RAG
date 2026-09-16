from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
cfg=yaml.safe_load((ROOT/'configs/baseline.yaml').read_text())
c=cfg['runners']['SPRINT']
assert cfg['methods']['SPRINT']['table_group'] == 'secondary'
assert c['historical_days'] == 28
assert c['day_steps'] == 288
assert c['use_vendor_native_defaults'] is True
for key in ['ranks','max_iter','smoke_max_iter','hyp_prior_list','hy1_list','batch_per','batch_iter','smoke_batch_per','smoke_batch_iter']:
    assert key not in c

s=(ROOT/'baseline/runners/sprint_runner.py').read_text()
h=(ROOT/'baseline/sprint_causal_history.py').read_text()
assert 'class SprintCanonicalHistory' in h
assert 'def resolve_canonical_history_source' in h
assert 'def sprint_vendor_defaults' in s
assert 'resolve_sprint_batch_schedule' not in s
assert 'pseudo_days' not in s and 'pseudo_days' not in h
assert "'algorithm_policy': 'pinned_vendor_native_defaults'" in s
assert 'missing_rate=0.0' in s
assert "'vendor_algorithm_defaults': native" in s
print('SPRINT VENDOR-NATIVE DEFAULTS SMOKE PASS')

for rel in ['pbs/methods/trc_development.pbs','pbs/methods/formal_secondary_patchstg.pbs']:
    text=(ROOT/rel).read_text()
    assert 'check_sprint_runtime.py' in text
    assert '--stage ' in text
print('SPRINT PBS RUNTIME CHECK WIRING PASS')
