"""Regression: Gadi Speed/Flow 182373268/182373271, C12 epoch 1.

The C1 gain classifier must train with BCEWithLogits while automatic mixed
precision is active. These tests exercise real C1/C2 forward and its loss.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from test_v5 import data, model
from tcrr_stla.v5.losses import compute_losses


def _batch_and_model():
    net = model()
    examples = data()
    # Ensure there are positive and negative training-label positions.
    torch.manual_seed(20261028)
    target = torch.randn(examples['x'].shape[0], 12, examples['x'].shape[-1])
    batch = {
        'y_norm': target,
        'y_raw': target,
        'future_valid': torch.ones_like(target, dtype=torch.bool),
    }
    return net, examples, batch


def test_c1_gain_logits_match_reported_probability():
    net, examples, _ = _batch_and_model()
    with torch.no_grad():
        out = net(**examples, mode='M11')
    assert out['cf_gain_logit'].shape == out['cf_gain_probability'].shape
    assert torch.allclose(
        out['cf_gain_probability'].float(),
        torch.sigmoid(out['cf_gain_logit'].float()), atol=2e-4, rtol=2e-4,
    )


@pytest.mark.parametrize('phase,mode', [('c12', 'M11'), ('joint', 'M111')])
def test_full_v5_loss_under_cpu_autocast_backward(phase, mode):
    net, examples, batch = _batch_and_model()
    net.set_stage(phase)
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        out = net(**examples, mode=mode,
                  shuffled_negatives=(phase == 'c12'))
        loss, stats = compute_losses(out, batch, 0., 1.,
                                     {'training': {}, 'v5': {}}, phase)
    assert torch.isfinite(loss)
    assert torch.isfinite(stats['cf_gain_bce'])
    loss.backward()
    grad = net.c1.gain_probability.weight.grad
    assert grad is not None and torch.isfinite(grad).all()


def test_only_amp_safe_bce_in_v5_loss():
    # Guard against re-introducing the original unsafe operation.
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] /
           'tcrr_stla' / 'v5' / 'losses.py').read_text()
    assert 'F.binary_cross_entropy(' not in src
    assert "o['cf_gain_logit']" in src
    assert 'F.binary_cross_entropy_with_logits(' in src


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires CUDA GPU')
def test_real_cuda_amp_c12_first_step():
    net, examples, batch = _batch_and_model()
    net = net.cuda()
    examples = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                for k, v in examples.items()}
    batch = {k: v.cuda() for k, v in batch.items()}
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        out = net(**examples, mode='M11')
        loss, stats = compute_losses(out, batch, 0., 1.,
                                     {'training': {}, 'v5': {}}, 'c12')
    assert torch.isfinite(loss)
    loss.backward()
    assert net.c1.gain_probability.weight.grad is not None
