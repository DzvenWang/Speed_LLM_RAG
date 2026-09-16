from __future__ import annotations

"""Vectorized compatibility implementation of the official LRTC-TMCP update.

The official repository publishes the method in ``lrtc-tmcp-demo.ipynb``.  Its
core algorithm is a three-mode ADMM-style low-rank tensor completion loop:
mode unfolding -> truncated MCP singular-value shrinkage -> fold/average ->
dual update.  This module preserves those update equations while vectorizing
independent TraffiDent H12 queries over a leading batch dimension.

The benchmark adapter deliberately differs only in two safety details:
1) an explicit observation mask is used instead of ``sparse_tensor == 0`` so
   legitimate traffic zeros cannot be mistaken for missing values;
2) no dense ground truth is passed into the completion routine, because the
   official notebook uses dense values only to print imputation metrics.
"""

import math
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class TMCPInfo:
    iterations: int
    converged_fraction: float
    max_relative_change: float


def _unfold(x: torch.Tensor, mode: int) -> torch.Tensor:
    """x: [B,D0,D1,D2] -> [B,D_mode,prod(other)]."""
    if x.ndim != 4 or mode not in (0, 1, 2):
        raise ValueError(f'Expected [B,D0,D1,D2], mode 0..2; got {tuple(x.shape)}, mode={mode}')
    y = torch.movedim(x, mode + 1, 1)
    return y.reshape(y.shape[0], y.shape[1], -1)


def _fold(mat: torch.Tensor, sample_shape: tuple[int, int, int], mode: int) -> torch.Tensor:
    others = [sample_shape[i] for i in range(3) if i != mode]
    y = mat.reshape(mat.shape[0], sample_shape[mode], *others)
    return torch.movedim(y, 1, mode + 1)


def _firm_shrinkage(s: torch.Tensor, lam: float, gamma: float) -> torch.Tensor:
    if gamma <= 1:
        return torch.where(s > lam, s, torch.zeros_like(s))
    out = gamma / (gamma - 1.0) * torch.relu(s - lam)
    return torch.where(s > gamma * lam, s, out)


def _svt_tmcp(mat: torch.Tensor, tau: float, theta: torch.Tensor | float, gamma: float) -> torch.Tensor:
    # torch.linalg.svd natively supports a leading batch dimension.
    u, s, vh = torch.linalg.svd(mat, full_matrices=False)
    shrunk = _firm_shrinkage(s, float(tau), float(gamma))
    if torch.is_tensor(theta):
        th = theta.to(device=s.device, dtype=s.dtype).reshape(-1)
        if th.numel() != s.shape[0]:
            raise ValueError('theta batch size mismatch')
        r = torch.ceil(th * s.shape[-1]).long().clamp_(0, s.shape[-1])
        keep = torch.arange(s.shape[-1], device=s.device)[None, :] < r[:, None]
        ss = torch.where(keep, s, shrunk)
    else:
        r = int(math.ceil(float(theta) * int(s.shape[-1])))
        r = min(max(r, 0), int(s.shape[-1]))
        keep = torch.arange(s.shape[-1], device=s.device)[None, :] < r
        ss = torch.where(keep, s, shrunk)
    # Batched reconstruction without materialising diag matrices.
    return (u * ss.unsqueeze(-2)) @ vh


@torch.no_grad()
def recover_batch(
    sparse_tensor: torch.Tensor,
    observed_mask: torch.Tensor,
    *,
    rho: float = 1e-5,
    theta0: float = 0.20,
    beta: float = 2.5,
    gamma: float = 20.0,
    factor: float = 1.05,
    tol: float = 1e-4,
    max_iter: int = 200,
    clamp_min: float = 0.0,
    clamp_max: float = 1e4,
) -> tuple[torch.Tensor, TMCPInfo]:
    """Recover a batch of 3-D tensors using the official TMCP update equations.

    Args:
        sparse_tensor: [B,D0,D1,D2], missing entries can contain any sentinel.
        observed_mask: same shape, True exactly where observations are available.

    Observed values are enforced after every update.  Each batch element is
    independent; batching is only a computational vectorization.
    """
    if sparse_tensor.ndim != 4:
        raise ValueError(f'LRTC-TMCP expects [B,D0,D1,D2], got {tuple(sparse_tensor.shape)}')
    if observed_mask.shape != sparse_tensor.shape:
        raise ValueError('observed_mask shape mismatch')
    x = sparse_tensor.float()
    obs = observed_mask.bool()
    B = x.shape[0]
    sample_shape = tuple(int(v) for v in x.shape[1:])
    dim = 3
    alpha = 1.0 / 3.0

    # Official initialization: fill missing entries with the mean of observed
    # entries.  Do it independently per query to prevent cross-query sharing.
    obs_f = obs.float()
    denom = obs_f.flatten(1).sum(1).clamp_min(1.0)
    means = (x * obs_f).flatten(1).sum(1) / denom
    M = torch.where(obs, x, means[:, None, None, None]).clone()
    Z = torch.zeros((dim, *x.shape), dtype=x.dtype, device=x.device)
    T = torch.zeros_like(Z)
    rho_t = float(rho)
    # Official notebook adapts theta to the missing fraction:
    # theta = theta0 * exp(-beta * missing_rate).  Compute it per query.
    missing_fraction = (~obs).float().flatten(1).mean(1)
    theta = float(theta0) * torch.exp(-float(beta) * missing_fraction)
    rel = torch.full((B,), float('inf'), dtype=x.dtype, device=x.device)

    for it in range(int(max_iter)):
        latest = M.clone()
        rho_t = float(min(max(rho_t * float(factor), 1e-10), 1e5))
        for k in range(dim):
            mk = _unfold(M - T[k] / rho_t, k)
            zk = _svt_tmcp(mk, alpha / rho_t, theta, float(gamma))
            Z[k] = _fold(zk, sample_shape, k)
        proposal = torch.mean(Z + T / rho_t, dim=0)
        M = torch.where(obs, x, proposal)
        M.clamp_(min=float(clamp_min), max=float(clamp_max))
        # Re-enforce observations after clamping, including legitimate zeros.
        M = torch.where(obs, x, M)
        T += rho_t * (Z - M.unsqueeze(0))
        num = torch.linalg.vector_norm((M - latest).flatten(1), dim=1)
        den = torch.linalg.vector_norm(latest.flatten(1), dim=1).clamp_min(1e-12)
        rel = num / den
        if bool(torch.all(rel < float(tol))):
            break

    info = TMCPInfo(
        iterations=it + 1,
        converged_fraction=float((rel < float(tol)).float().mean().item()),
        max_relative_change=float(rel.max().item()),
    )
    return M, info
