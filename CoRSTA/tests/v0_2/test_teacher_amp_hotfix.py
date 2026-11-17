"""Regression for the Gadi Speed/Flow FP16 teacher decomposition failure."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pytest
import torch
from torch import nn
from corsta_v2.counterfactual import teacher as teacher_module


class NoopSTLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(()))


@pytest.mark.parametrize('scale', [1.0, 10.0, 100.0])
def test_teacher_decomposition_with_half_precision_predictions(monkeypatch, scale):
    torch.manual_seed(7)
    batch_size, horizon, nodes, length, hidden = 8, 12, 388, 12, 64
    # The old implementation performed cf = pf - pm in FP16, which routinely
    # gives 1e-3--1e-1 reconstruction errors for these scales.
    pf = (torch.randn(batch_size, horizon, nodes) * scale).half()
    pm = (torch.randn(batch_size, horizon, nodes) * scale).half()
    hf = torch.randn(batch_size, length, nodes, hidden).half()
    hm = torch.randn(batch_size, length, nodes, hidden).half()

    def fake_base_features(stla, x, mask, tod, dow):
        return (hf, None, pf) if bool((x == 1).all()) else (hm, None, pm)

    monkeypatch.setattr(teacher_module, 'base_features', fake_base_features)
    b = {
        'x_full': torch.ones(batch_size, length, nodes),
        'original_valid': torch.ones(batch_size, length, nodes),
        'x': torch.zeros(batch_size, length, nodes),
        'mask': torch.zeros(batch_size, length, nodes),
        'tod': torch.zeros(batch_size, length),
        'dow': torch.zeros(batch_size, length),
        'y_norm': torch.randn(batch_size, horizon, nodes) * scale,
        'future_valid': torch.ones(batch_size, horizon, nodes, dtype=torch.bool),
    }
    out = teacher_module.PairedTeacher(NoopSTLA())(b, hm, pm)
    assert out['cf_target'].dtype == torch.float32
    assert out['residual_target'].dtype == torch.float32
    assert out['latent_cf_target'].dtype == torch.float32
    delta = (out['cf_target'] + out['residual_target'] -
             (b['y_norm'] - pm.float())).abs()
    eps = torch.finfo(torch.float32).eps
    tolerance = 2e-6 + 8*eps * (pf.float().abs()+pm.float().abs()+b['y_norm'].abs())
    assert torch.all(delta <= tolerance)


def test_teacher_rejects_nonfinite_prediction(monkeypatch):
    pf = torch.tensor([[[float('nan')]]], dtype=torch.float16)
    def fake(*args, **kwargs):
        return torch.zeros((1,1,1,2)), None, pf
    monkeypatch.setattr(teacher_module, 'base_features', fake)
    b = dict(x_full=torch.ones((1,1,1)), original_valid=torch.ones((1,1,1)),
             x=torch.zeros((1,1,1)),mask=torch.zeros((1,1,1)),
             tod=torch.zeros((1,1)),dow=torch.zeros((1,1)),
             y_norm=torch.zeros((1,1,1)),
             future_valid=torch.ones((1,1,1),dtype=torch.bool))
    with pytest.raises(RuntimeError,match='non-finite p_full'):
        teacher_module.PairedTeacher(NoopSTLA())(b)


def test_teacher_ignores_invalid_future_label_nan(monkeypatch):
    pf = torch.tensor([[[1.0,2.0]]]).half()
    def fake(*args, **kwargs):
        return torch.zeros((1,1,2,2)), None, pf
    monkeypatch.setattr(teacher_module, 'base_features', fake)
    b = dict(x_full=torch.ones((1,1,2)), original_valid=torch.ones((1,1,2)),
             x=torch.zeros((1,1,2)), mask=torch.zeros((1,1,2)),
             tod=torch.zeros((1,1)),dow=torch.zeros((1,1)),
             y_norm=torch.tensor([[[3.0,float('nan')]]]),
             future_valid=torch.tensor([[[True,False]]]))
    out=teacher_module.PairedTeacher(NoopSTLA())(b)
    assert torch.isfinite(out['cf_target']).all()
    assert torch.isfinite(out['residual_target']).all()
