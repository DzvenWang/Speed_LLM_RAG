"""Regression for the v9.3 Gadi AMP overflow after residual Epoch 1."""
import pytest
import torch
from tcrr_stla.v93.training import _safe_optimizer_step


def test_gradscaler_skips_overflow_and_recovers():
    # CPU GradScaler exposes the same public step/unscale/update semantics as CUDA.
    parameter = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float32))
    optimizer = torch.optim.SGD([parameter], lr=0.05)
    scaler = torch.amp.GradScaler('cpu', enabled=True, init_scale=65536.0,
                                  growth_interval=2000)
    before_weight = parameter.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    scaler.scale((parameter - 1).square().sum()).backward()
    parameter.grad.fill_(float('inf'))  # simulate the Gadi overflow
    scaler.unscale_(optimizer)
    updated, old_scale, new_scale = _safe_optimizer_step(
        optimizer, scaler, [parameter], 5.0, 'residual', 0)
    assert not updated
    assert new_scale < old_scale
    torch.testing.assert_close(parameter.detach(), before_weight)

    optimizer.zero_grad(set_to_none=True)
    scaler.scale((parameter - 1).square().sum()).backward()
    scaler.unscale_(optimizer)
    updated, old_scale, new_scale = _safe_optimizer_step(
        optimizer, scaler, [parameter], 5.0, 'residual', 1)
    assert updated
    assert new_scale >= old_scale
    assert not torch.equal(parameter.detach(), before_weight)


def test_fp32_nonfinite_gradient_still_fails():
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.05)
    parameter.grad = torch.tensor([float('inf')])
    with pytest.raises(RuntimeError, match='Non-finite gradient.*FP32'):
        _safe_optimizer_step(optimizer, None, [parameter], 5.0, 'residual', 7)
