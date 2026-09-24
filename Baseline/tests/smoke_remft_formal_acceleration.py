#!/usr/bin/env python3
from pathlib import Path
import math
import yaml

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT / 'configs/baseline.yaml').read_text())
c = cfg['runners']['ReMFT']

assert int(c['batch_size']) * int(c['gradient_accumulation_steps']) == int(c['effective_batch_size'])
assert int(c['effective_batch_size']) % int(c['formal_batch_size']) == 0

assert int(c['formal_batch_size']) == 8, c['formal_batch_size']
assert int(c['formal_eval_batch_size']) == 8, c['formal_eval_batch_size']
assert int(c['effective_batch_size']) // int(c['formal_batch_size']) == 3
assert bool(c['formal_use_amp']) is True
assert bool(c.get('development_use_amp', False)) is False
expected = math.ceil(
    int(c['formal_reference_development_train_rows']) / int(c['effective_batch_size'])
) * int(c['formal_reference_development_best_epoch'])
assert int(c['formal_max_optimizer_updates']) == expected

src = (ROOT / 'baseline/runners/remft_runner.py').read_text()
for needle in [
    'REMFT TRAINING PLAN',
    'FORMAL UPDATE BUDGET REACHED',
    'torch.autocast',
    'normalize_bert_depth_to_checkpoint',
    'checkpoint_consistent_pretrained_bert_depth',
    'formal_max_optimizer_updates',
]:
    assert needle in src, needle

print('REMFT FORMAL ACCELERATION STATIC SMOKE PASS', {
    'effective_batch_size': int(c['effective_batch_size']),
    'formal_batch_size': int(c['formal_batch_size']),
    'formal_accumulation': int(c['effective_batch_size']) // int(c['formal_batch_size']),
    'formal_max_optimizer_updates': int(c['formal_max_optimizer_updates']),
})
