from __future__ import annotations
import torch


def make_inputs(x, tod_hist, dow_hist, device):
    x = x.to(device, non_blocking=True).float().unsqueeze(-1)
    B, T, N, _ = x.shape
    tod = tod_hist.to(device, non_blocking=True).long().unsqueeze(-1).expand(B, T, N)
    dow = dow_hist.to(device, non_blocking=True).long().unsqueeze(-1).expand(B, T, N)
    te = torch.stack([tod, dow], dim=-1)
    return x, te


def forward_norm(model, batch, device):
    x, te = make_inputs(batch["x_obs"], batch["tod_hist"], batch["dow_hist"], device)
    pred = model(x, te)
    if pred.ndim == 4 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
    return pred


def masked_mae(pred, target, mask):
    m = mask.bool() & torch.isfinite(target) & torch.isfinite(pred)
    if not torch.any(m):
        return pred.sum() * 0.0
    return torch.abs(pred - target)[m].mean()
