#!/usr/bin/env python3
from __future__ import annotations

import ast
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'baseline/runners/remft_runner.py'


def load_optimizer_step():
    source = RUNNER.read_text()
    tree = ast.parse(source)
    fn = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_optimizer_step'),
        None,
    )
    if fn is None:
        raise AssertionError('_optimizer_step not found')
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {'torch': torch, 'nn': nn}
    exec(compile(module, str(RUNNER), 'exec'), ns, ns)
    return ns['_optimizer_step'], source


class FakeScaler:
    def __init__(self, *, scale: float = 65536.0, overflow: bool = False):
        self.scale = float(scale)
        self.overflow = bool(overflow)

    def unscale_(self, optimizer):
        return None

    def get_scale(self):
        return self.scale

    def step(self, optimizer):
        if not self.overflow:
            optimizer.step()

    def update(self):
        if self.overflow:
            self.scale *= 0.5


def main():
    step_fn, source = load_optimizer_step()

    # AMP overflow must be treated as a skipped step, not a fatal error.
    p = nn.Parameter(torch.tensor([1.0]))
    opt = torch.optim.SGD([p], lr=0.1)
    p.grad = torch.tensor([float('inf')])
    scaler = FakeScaler(overflow=True)
    before = p.detach().clone()
    meta = step_fn(
        opt=opt,
        scaler=scaler,
        trainable=[p],
        gradient_clip=1.0,
        use_amp=True,
    )
    assert meta['skipped'] is True, meta
    assert meta['grad_finite'] is False, meta
    assert meta['scale_after'] < meta['scale_before'], meta
    assert torch.equal(before, p.detach()), 'overflowed AMP step must not update parameters'

    # A finite AMP step must count as a real optimizer update.
    p2 = nn.Parameter(torch.tensor([1.0]))
    opt2 = torch.optim.SGD([p2], lr=0.1)
    p2.grad = torch.tensor([1.0])
    scaler2 = FakeScaler(overflow=False)
    before2 = p2.detach().clone()
    meta2 = step_fn(
        opt=opt2,
        scaler=scaler2,
        trainable=[p2],
        gradient_clip=1.0,
        use_amp=True,
    )
    assert meta2['skipped'] is False, meta2
    assert meta2['grad_finite'] is True, meta2
    assert not torch.equal(before2, p2.detach()), 'finite AMP step should update parameters'

    # FP32 non-finite gradients remain fatal.
    p3 = nn.Parameter(torch.tensor([1.0]))
    opt3 = torch.optim.SGD([p3], lr=0.1)
    p3.grad = torch.tensor([float('inf')])
    try:
        step_fn(
            opt=opt3,
            scaler=FakeScaler(overflow=False),
            trainable=[p3],
            gradient_clip=1.0,
            use_amp=False,
        )
    except RuntimeError as exc:
        assert 'non-finite FP32 grad norm' in str(exc)
    else:
        raise AssertionError('FP32 non-finite gradient must remain fatal')

    assert 'amp_skipped_steps += 1' in source
    assert "if step_meta['skipped']" in source
    assert 'REMFT AMP SKIP' in source
    assert "'amp_skipped_steps': amp_skipped_steps" in source
    assert 'ReMFT non-finite grad norm epoch=' not in source

    print(
        'REMFT AMP OVERFLOW SMOKE PASS',
        {
            'overflow_skipped': meta['skipped'],
            'scale_before': meta['scale_before'],
            'scale_after': meta['scale_after'],
            'finite_step_skipped': meta2['skipped'],
        },
    )


if __name__ == '__main__':
    main()
