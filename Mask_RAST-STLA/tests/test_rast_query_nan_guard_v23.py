"""Regression: unchanged retrieved candidate values must not poison gradients.

Production incident 181031484: non-finite temporal query encoding in all 64
rows during the 7th epoch. Zero weighted variance + sqrt(0) is a deterministic
bad backward path in the previous risk-disagreement feature.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch

from mask_rast_stla.model import stable_candidate_disagreement


def test_exact_zero_variance_has_finite_backward():
    torch.manual_seed(20261012)
    b, k, h, n = 2, 5, 12, 9
    # All future residuals are identical (including fully masked/all-none).
    weights = torch.full((b,k,h,n),0.2,requires_grad=True)
    residuals = torch.zeros(b,k,h,n,requires_grad=True)
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        disagreement=stable_candidate_disagreement(weights,residuals)
        loss=disagreement.sum()
    assert disagreement.shape==(b,h,n)
    assert torch.isfinite(disagreement).all()
    assert torch.allclose(disagreement,torch.zeros_like(disagreement),atol=1e-6)
    loss.backward()
    assert weights.grad is not None and torch.isfinite(weights.grad).all()
    assert residuals.grad is not None and torch.isfinite(residuals.grad).all()


def test_all_none_and_soft_candidates_fp32_safe():
    weights=torch.zeros((1,4,3,6),requires_grad=True)
    residuals=torch.randn((1,4,3,6),requires_grad=True)
    out=stable_candidate_disagreement(weights,residuals)
    assert (out==0).all()
    out.sum().backward()
    assert torch.isfinite(weights.grad).all()
    assert torch.isfinite(residuals.grad).all()


def test_weighted_variance_agrees_with_reference_away_from_zero():
    torch.manual_seed(3)
    w0=torch.softmax(torch.randn(2,3,4,5),dim=1)
    r=torch.randn(2,3,4,5)
    expected=(w0*(r-(w0*r).sum(1,keepdim=True)).square()).sum(1).sqrt()
    actual=stable_candidate_disagreement(w0,r)
    assert torch.allclose(actual,expected,atol=1.2e-3,rtol=1e-2)


def test_key_normalization_has_finite_gradient_near_zero():
    from mask_rast_stla.encoders import StateEncoder
    enc=StateEncoder(7,12,graph_k=3)
    # Validate no gradient explosion from normalization of near-zero learned keys.
    with torch.no_grad():
        enc.temporal_key.weight.zero_()
        enc.temporal_key.bias.zero_()
        enc.spatial_key.weight.zero_()
        enc.spatial_key.bias.zero_()
    x=torch.randn(2,12,7)
    mask=torch.ones_like(x)
    tod=torch.zeros(2,12,dtype=torch.long)
    encoded=enc(x,mask,tod,tod)
    loss=(encoded['temporal_key']+encoded['spatial_key']).sum()
    loss.backward()
    assert torch.isfinite(enc.temporal_key.weight.grad).all()
    assert torch.isfinite(enc.spatial_key.weight.grad).all()
