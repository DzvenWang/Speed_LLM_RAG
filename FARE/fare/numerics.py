"""Numerical safety primitives for CUDA FP16 autocast.

Ranking logits are small relative to evidence tensors; computing all masks,
Top-K ranking and softmax inputs in float32 prevents illegal negative sentinels
from overflowing torch.float16 (-65504 is its lowest finite value).
"""
import torch


def mask_invalid_logits(logits: torch.Tensor, invalid: torch.Tensor,
                        value: float = -1e7) -> torch.Tensor:
    """Return float32 logits with invalid entries suppressed.

    The explicit float32 cast is intentional even with autocast enabled.  Do
    not replace it with ``logits.masked_fill`` or a half-precision sentinel.
    """
    return logits.float().masked_fill(invalid, value)
