from pathlib import Path
import ast,yaml
ROOT=Path(__file__).resolve().parents[1]
cfg=yaml.safe_load((ROOT/'configs/baseline.yaml').read_text())
runner=(ROOT/'baseline/runners/mogernn_runner.py').read_text()
checker=(ROOT/'scripts/check_mogernn_inference_contract.py').read_text()
ast.parse(runner); ast.parse(checker)
assert 'def assert_eval_target_independence' in runner
assert "'zeros':torch.zeros_like(y)" in runner
assert "'time_reversed':torch.flip(y,dims=[1])" in runner
assert 'Do not use this result; reproduce the pinned vendor inference path instead.' in runner
assert cfg['runners']['MoGERNN']['require_eval_target_independence'] is True
assert float(cfg['runners']['MoGERNN']['eval_target_independence_atol']) == 1e-6
assert cfg['runners']['MoGERNN']['max_epochs'] == 50
assert cfg['runners']['MoGERNN']['patience'] == 3
text=(ROOT/'pbs/methods/trc_development.pbs').read_text()
assert 'check_mogernn_inference_contract.py' in text
print('MOGERNN INFERENCE CONTRACT STATIC SMOKE PASS')
