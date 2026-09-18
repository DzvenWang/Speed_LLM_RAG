from __future__ import annotations

"""Vectorized compatibility implementation of the official LRTC-TMCP update.

The official repository publishes the method in ``lrtc-tmcp-demo.ipynb``. Its
core algorithm is a three-mode ADMM-style low-rank tensor completion loop:
mode unfolding -> truncated MCP singular-value shrinkage -> fold/average ->
dual update. This module preserves those equations while vectorizing
independent TraffiDent queries over a leading batch dimension.

Benchmark compatibility details:
1) an explicit observation mask is used instead of ``sparse_tensor == 0`` so
   legitimate traffic zeros cannot be mistaken for missing values;
2) no dense ground truth is passed into completion;
3) exact Gram-eigendecomposition may replace direct SVD. Mathematically it
   reconstructs the same singular-value transform while diagonalising only the
   smaller Gram matrix. This is an implementation optimisation, not a model
   change;
4) already-converged queries are removed from later iterations. Each query is
   independent, so this is equivalent to terminating that query at the same
   tolerance in a standalone run.
"""

import math
import time
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TMCPInfo:
    iterations: int
    converged_fraction: float
    max_relative_change: float
    mean_query_iterations: float
    max_query_iterations: int
    min_query_iterations: int
    elapsed_seconds: float
    decomposition: str


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


def _select_transformed_singular_values(
    s: torch.Tensor,
    tau: float,
    theta: torch.Tensor | float,
    gamma: float,
) -> torch.Tensor:
    shrunk = _firm_shrinkage(s, float(tau), float(gamma))
    rank = int(s.shape[-1])
    if torch.is_tensor(theta):
        th = theta.to(device=s.device, dtype=s.dtype).reshape(-1)
        if th.numel() != s.shape[0]:
            raise ValueError('theta batch size mismatch')
        r = torch.ceil(th * rank).long().clamp_(0, rank)
        keep = torch.arange(rank, device=s.device)[None, :] < r[:, None]
    else:
        r = int(math.ceil(float(theta) * rank))
        r = min(max(r, 0), rank)
        keep = torch.arange(rank, device=s.device)[None, :] < r
    return torch.where(keep, s, shrunk)


def _svt_tmcp_direct_svd(
    mat: torch.Tensor,
    tau: float,
    theta: torch.Tensor | float,
    gamma: float,
) -> torch.Tensor:
    """Reference implementation used by equivalence tests."""
    u, s, vh = torch.linalg.svd(mat, full_matrices=False)
    ss = _select_transformed_singular_values(s, tau, theta, gamma)
    return (u * ss.unsqueeze(-2)) @ vh


def _svt_tmcp_gram_eigh(
    mat: torch.Tensor,
    tau: float,
    theta: torch.Tensor | float,
    gamma: float,
) -> torch.Tensor:
    """Apply the same singular-value transform using the smaller Gram matrix.

    For A in R^(m x n):
      m <= n: diagonalise A A^T and return U f(S)/S U^T A.
      n <  m: diagonalise A^T A and return A V f(S)/S V^T.

    ``torch.linalg.eigh`` returns eigenvalues in ascending order, hence the
    explicit reversal. Negative round-off eigenvalues are clamped to zero.
    """
    if mat.ndim != 3:
        raise ValueError(f'Expected batched matrix [B,m,n], got {tuple(mat.shape)}')
    _, m, n = mat.shape
    eps = torch.finfo(mat.dtype).eps

    if m <= n:
        gram = mat @ mat.transpose(-1, -2)  # [B,m,m]
        evals, vecs = torch.linalg.eigh(gram)
        evals = evals.flip(-1).clamp_min_(0.0)
        vecs = vecs.flip(-1)
        s = torch.sqrt(evals)
        ss = _select_transformed_singular_values(s, tau, theta, gamma)
        ratio = torch.where(s > eps, ss / s, torch.zeros_like(s))
        # U diag(ratio) U^T A
        return (vecs * ratio.unsqueeze(-2)) @ (vecs.transpose(-1, -2) @ mat)

    gram = mat.transpose(-1, -2) @ mat  # [B,n,n]
    evals, vecs = torch.linalg.eigh(gram)
    evals = evals.flip(-1).clamp_min_(0.0)
    vecs = vecs.flip(-1)
    s = torch.sqrt(evals)
    ss = _select_transformed_singular_values(s, tau, theta, gamma)
    ratio = torch.where(s > eps, ss / s, torch.zeros_like(s))
    # A V diag(ratio) V^T
    return ((mat @ vecs) * ratio.unsqueeze(-2)) @ vecs.transpose(-1, -2)


def _svt_tmcp(
    mat: torch.Tensor,
    tau: float,
    theta: torch.Tensor | float,
    gamma: float,
    *,
    decomposition: str,
) -> torch.Tensor:
    if decomposition == 'gram_eigh':
        return _svt_tmcp_gram_eigh(mat, tau, theta, gamma)
    if decomposition == 'direct_svd':
        return _svt_tmcp_direct_svd(mat, tau, theta, gamma)
    raise ValueError(f'Unknown LRTC-TMCP decomposition={decomposition!r}')


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
    decomposition: str = 'gram_eigh',
    active_convergence: bool = True,
    progress_every: int = 0,
    progress_prefix: str = 'LRTC-TMCP',
) -> tuple[torch.Tensor, TMCPInfo]:
    """Recover a batch of 3-D tensors using the official TMCP update equations.

    Args:
        sparse_tensor: [B,D0,D1,D2], missing entries can contain any sentinel.
        observed_mask: same shape, True exactly where observations are available.
        decomposition: ``gram_eigh`` (exact smaller-Gram implementation) or
            ``direct_svd`` (reference implementation).
        active_convergence: if true, queries satisfying ``tol`` stop receiving
            updates while unconverged queries continue independently.
        progress_every: print an iteration-level status every N iterations; 0
            disables iteration logging.

    Observed values are enforced after every update. Each batch element is
    independent; batching is only a computational vectorization.
    """
    if sparse_tensor.ndim != 4:
        raise ValueError(f'LRTC-TMCP expects [B,D0,D1,D2], got {tuple(sparse_tensor.shape)}')
    if observed_mask.shape != sparse_tensor.shape:
        raise ValueError('observed_mask shape mismatch')
    if int(max_iter) < 1:
        raise ValueError('max_iter must be >= 1')

    t0 = time.perf_counter()
    x = sparse_tensor.float()
    obs = observed_mask.bool()
    B = x.shape[0]
    sample_shape = tuple(int(v) for v in x.shape[1:])
    dim = 3
    alpha = 1.0 / 3.0

    # Official initialization: fill missing entries with the per-query mean of
    # observed entries. No cross-query sharing is permitted.
    obs_f = obs.float()
    denom = obs_f.flatten(1).sum(1).clamp_min(1.0)
    means = (x * obs_f).flatten(1).sum(1) / denom
    M = torch.where(obs, x, means[:, None, None, None]).clone()
    Z = torch.zeros((dim, *x.shape), dtype=x.dtype, device=x.device)
    T = torch.zeros_like(Z)
    rho_t = float(rho)

    # Official notebook adapts theta to missing fraction.
    missing_fraction = (~obs).float().flatten(1).mean(1)
    theta = float(theta0) * torch.exp(-float(beta) * missing_fraction)
    rel = torch.full((B,), float('inf'), dtype=x.dtype, device=x.device)
    active = torch.ones((B,), dtype=torch.bool, device=x.device)
    query_iterations = torch.zeros((B,), dtype=torch.int32, device=x.device)

    if progress_every:
        print(
            f'{progress_prefix} START batch={B} shape={sample_shape} '
            f'decomposition={decomposition} active_convergence={bool(active_convergence)} '
            f'max_iter={int(max_iter)} tol={float(tol):.3e}',
            flush=True,
        )

    for it in range(int(max_iter)):
        if active_convergence:
            active_idx = torch.nonzero(active, as_tuple=False).flatten()
        else:
            active_idx = torch.arange(B, device=x.device)
        if active_idx.numel() == 0:
            break

        latest = M[active_idx].clone()
        rho_t = float(min(max(rho_t * float(factor), 1e-10), 1e5))
        theta_active = theta[active_idx]

        for k in range(dim):
            mk = _unfold(M[active_idx] - T[k, active_idx] / rho_t, k)
            zk = _svt_tmcp(
                mk,
                alpha / rho_t,
                theta_active,
                float(gamma),
                decomposition=decomposition,
            )
            Z[k, active_idx] = _fold(zk, sample_shape, k)

        proposal = torch.mean(Z[:, active_idx] + T[:, active_idx] / rho_t, dim=0)
        updated = torch.where(obs[active_idx], x[active_idx], proposal)
        updated.clamp_(min=float(clamp_min), max=float(clamp_max))
        updated = torch.where(obs[active_idx], x[active_idx], updated)
        M[active_idx] = updated
        T[:, active_idx] += rho_t * (Z[:, active_idx] - M[active_idx].unsqueeze(0))

        num = torch.linalg.vector_norm((M[active_idx] - latest).flatten(1), dim=1)
        den = torch.linalg.vector_norm(latest.flatten(1), dim=1).clamp_min(1e-12)
        rel_active = num / den
        rel[active_idx] = rel_active
        query_iterations[active_idx] += 1

        if active_convergence:
            converged_now = rel_active < float(tol)
            if bool(converged_now.any()):
                active[active_idx[converged_now]] = False
        else:
            active = rel >= float(tol)

        if progress_every and (
            (it + 1) == 1
            or (it + 1) % int(progress_every) == 0
            or not bool(active.any())
            or (it + 1) == int(max_iter)
        ):
            elapsed = time.perf_counter() - t0
            finite_rel = rel[torch.isfinite(rel)]
            max_rel = float(finite_rel.max().item()) if finite_rel.numel() else float('inf')
            print(
                f'{progress_prefix} ITER {it + 1}/{int(max_iter)} '
                f'active={int(active.sum().item())}/{B} '
                f'converged={B - int(active.sum().item())}/{B} '
                f'max_rel={max_rel:.6e} elapsed={elapsed:.2f}s',
                flush=True,
            )

        if not bool(active.any()):
            break

    elapsed = time.perf_counter() - t0
    converged = rel < float(tol)
    info = TMCPInfo(
        iterations=int(query_iterations.max().item()),
        converged_fraction=float(converged.float().mean().item()),
        max_relative_change=float(rel.max().item()),
        mean_query_iterations=float(query_iterations.float().mean().item()),
        max_query_iterations=int(query_iterations.max().item()),
        min_query_iterations=int(query_iterations.min().item()),
        elapsed_seconds=float(elapsed),
        decomposition=str(decomposition),
    )
    return M, info
