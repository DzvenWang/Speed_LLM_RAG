"""Regression for Gadi 2026-10-12 failure: risk BCE inside autocast."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tcrr_stla.config import load_config
from tcrr_stla.model import TCRRSTLA
from tcrr_stla.trainer import compute_losses
from test_tcrr_v2 import STLA


def test_amp_risk_logit_loss_backward():
    torch.manual_seed(1001)
    B, H, P, N, K = 2, 12, 12, 7, 6
    legacy_cfg=load_config();legacy_cfg['version']='2.0.0'
    model = TCRRSTLA(STLA(), legacy_cfg)
    x = torch.randn(B, H, N)
    mask = (torch.rand(B, H, N) > 0.4).float()
    td = torch.zeros(B, H, dtype=torch.long)
    ev = dict(candidate_y=torch.randn(B, K, P, N), residual=torch.randn(B, K, P, N)*0.2,
              candidate_scores=torch.randn(B, K), candidate_coverage=torch.rand(B, K),
              hist_distance=torch.rand(B, K, N))
    y = torch.randn(B, P, N)
    batch = dict(y_norm=y, y_raw=y, future_valid=torch.ones(B, P, N, dtype=torch.bool))
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        out = model(x, mask, td, td, **ev)
        loss, logs = compute_losses(out, batch, 0., 1., legacy_cfg, 'full')
    assert torch.isfinite(loss)
    assert abs(logs['risk']) < 1e4
    assert torch.allclose(out['risk_probability'].float(), torch.sigmoid(out['risk_logit'].float()), atol=2e-3)
    loss.backward()
    grad = model.risk_gate[-1].weight.grad
    assert grad is not None and torch.isfinite(grad).all()
