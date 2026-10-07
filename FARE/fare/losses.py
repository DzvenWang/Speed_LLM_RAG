from __future__ import annotations
import torch


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.bool()
    if not torch.any(m):
        return (pred - target).abs().mean() * 0.0
    return (pred - target).abs()[m].mean()


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.bool()
    if not torch.any(m):
        return ((pred - target) ** 2).mean() * 0.0
    return ((pred - target) ** 2)[m].mean()
