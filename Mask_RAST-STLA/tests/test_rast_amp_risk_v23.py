"""Regression for Gadi 2026-10-12 failure: risk BCE inside autocast."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from mask_rast_stla.config import load_config
from mask_rast_stla.model import MaskRASTSTLA
from mask_rast_stla.memory import build_memory_index
from mask_rast_stla.trainer import losses
from test_rast_v2 import STLA
from test_rast_v23 import make_repo


def test_amp_risk_logit_loss_backward(tmp_path):
    torch.manual_seed(1002)
    B, H, P, N = 2, 12, 12, 7
    cfg = load_config()
    cfg['retrieval'].update(top_k=2, regional_top_k=2, graph_neighbors=3)
    model = MaskRASTSTLA(STLA(), N, H, P, cfg)
    model.set_graph(torch.tensor([[(i+j+1)%N for j in range(3)] for i in range(N)]))
    repo = make_repo(tmp_path, M=35, H=H, P=P, N=N)
    index = build_memory_index(model.memory_encoder, repo, 'cpu', 12)
    x = torch.randn(B,H,N)
    mask = (torch.rand(B,H,N) > .4).float()
    td = torch.zeros(B,H,dtype=torch.long)
    y = torch.randn(B,P,N)
    batch = dict(x=x, mask=mask, tod=td, dow=td,
                 window_start=torch.tensor([2000, 2200]), y_norm=y, y_raw=y,
                 future_valid=torch.ones(B,P,N,dtype=torch.bool))
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        out = model(batch, index, repo)
        loss, logs = losses(out, batch, 0., 1., cfg, 'full')
    assert torch.isfinite(loss)
    assert abs(logs['risk']) < 1e4
    assert torch.allclose(out['risk_probability'].float(), torch.sigmoid(out['risk_logit'].float()), atol=2e-3)
    loss.backward()
    grad = model.risk_gate[-1].weight.grad
    assert grad is not None and torch.isfinite(grad).all()
