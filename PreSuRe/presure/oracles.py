from __future__ import annotations
import numpy as np
import torch


def weighted_query_objective(pred, y, valid, direct, support, weights):
    """Weighted MAE objective per query/candidate.

    pred: [B,K,P,N] (K may be 1)
    y:    [B,1,P,N]
    valid:[B,1,P,N]
    direct/support: [B,N]
    returns [B,K]
    """
    m = valid.bool()
    e = torch.abs(pred - y)
    dm = direct[:, None, None, :].expand_as(m)
    sm = support[:, None, None, :].expand_as(m)

    def mae(mask):
        mm = m & mask
        den = mm.flatten(2).sum(-1).clamp_min(1)
        return (e * mm).flatten(2).sum(-1) / den

    one = torch.ones_like(m, dtype=torch.bool)
    return (
        float(weights['overall']) * mae(one)
        + float(weights['masked']) * mae(dm)
        + float(weights['support']) * mae(sm)
    )


def weighted_horizon_objective(pred, y, valid, direct, support, weights):
    """Weighted MAE per candidate and horizon, reducing over nodes only.

    pred: [B,K,P,N], returns [B,K,P].
    """
    m = valid.bool()
    e = torch.abs(pred - y)
    dm = direct[:, None, None, :].expand_as(m)
    sm = support[:, None, None, :].expand_as(m)

    def mae(mask):
        mm = m & mask
        den = mm.sum(-1).clamp_min(1)
        return (e * mm).sum(-1) / den

    one = torch.ones_like(m, dtype=torch.bool)
    return (
        float(weights['overall']) * mae(one)
        + float(weights['masked']) * mae(dm)
        + float(weights['support']) * mae(sm)
    )


def per_query_mae_np(pred, y, valid):
    v = np.asarray(valid, bool)
    den = v.sum((1, 2)).clip(min=1)
    return (np.abs(np.asarray(pred) - np.asarray(y)) * v).sum((1, 2)) / den


def relative_improvements(base_metrics, new_metrics):
    def rel(key):
        b = float(base_metrics[key]); n = float(new_metrics[key])
        return (b - n) / b if b != 0 else 0.0
    return {
        'overall': rel('overall_MAE'),
        'masked': rel('masked_MAE'),
        'support': rel('support_MAE'),
        'H12_overall': rel('H12_overall_MAE'),
        'H12_masked': rel('H12_masked_MAE'),
        'H12_support': rel('H12_support_MAE'),
    }


def oracle_global_batch(base, y, valid, direct, support, residuals, alphas, weights):
    """Single support + one global alpha per query.

    residuals: [B,K,P,N]. Returns final, local candidate index, alpha, objective gain.
    """
    b = base
    base_obj = weighted_query_objective(
        b[:, None], y[:, None], valid[:, None], direct, support, weights
    ).squeeze(1)
    best = base_obj.clone()
    best_idx = torch.full((len(b),), -1, dtype=torch.long, device=b.device)
    best_alpha = torch.zeros(len(b), dtype=b.dtype, device=b.device)
    for alpha in alphas:
        p = b[:, None] + float(alpha) * residuals
        o = weighted_query_objective(p, y[:, None], valid[:, None], direct, support, weights)
        val, idx = o.min(1)
        take = val < best
        best = torch.where(take, val, best)
        best_idx = torch.where(take, idx, best_idx)
        best_alpha = torch.where(take, torch.full_like(best_alpha, float(alpha)), best_alpha)
    final = b.clone()
    good = best_idx >= 0
    if torch.any(good):
        rr = residuals[good, best_idx[good]]
        final[good] = b[good] + best_alpha[good, None, None] * rr
    return final, best_idx, best_alpha, base_obj - best


def oracle_horizon_batch(base, y, valid, direct, support, residuals, alphas, weights):
    """Choose one support + alpha independently for each forecast horizon.

    This is diagnostic-only and uses query future. Returns final and [B,P] choices.
    """
    B, P, _ = base.shape
    base_obj = weighted_horizon_objective(
        base[:, None], y[:, None], valid[:, None], direct, support, weights
    ).squeeze(1)
    best = base_obj.clone()
    best_idx = torch.full((B, P), -1, dtype=torch.long, device=base.device)
    best_alpha = torch.zeros((B, P), dtype=base.dtype, device=base.device)
    for alpha in alphas:
        p = base[:, None] + float(alpha) * residuals
        o = weighted_horizon_objective(p, y[:, None], valid[:, None], direct, support, weights)
        val, idx = o.min(1)  # [B,P]
        take = val < best
        best = torch.where(take, val, best)
        best_idx = torch.where(take, idx, best_idx)
        best_alpha = torch.where(take, torch.full_like(best_alpha, float(alpha)), best_alpha)
    safe = best_idx.clamp_min(0)
    # residuals [B,K,P,N]; gather K independently for each P
    gather_idx = safe[:, None, :, None].expand(-1, 1, -1, residuals.shape[-1])
    selected = torch.gather(residuals, 1, gather_idx).squeeze(1)
    use = (best_idx >= 0).to(base.dtype)
    final = base + use[:, :, None] * best_alpha[:, :, None] * selected
    return final, best_idx, best_alpha, base_obj - best


def oracle_node_horizon_batch(base, y, valid, residuals, alphas):
    """Extremely loose node×horizon local routing oracle.

    Every valid (horizon,node) cell may choose a different candidate and alpha.
    It is *not deployable* and is only used to diagnose whether local transport
    has substantially more headroom than query-level residual transfer.
    """
    B, P, N = base.shape
    base_err = torch.abs(base - y)
    best_err = base_err.clone()
    best_idx = torch.full((B, P, N), -1, dtype=torch.long, device=base.device)
    best_alpha = torch.zeros((B, P, N), dtype=base.dtype, device=base.device)
    for alpha in alphas:
        err = torch.abs(base[:, None] + float(alpha) * residuals - y[:, None])
        val, idx = err.min(1)
        take = (val < best_err) & valid.bool()
        best_err = torch.where(take, val, best_err)
        best_idx = torch.where(take, idx, best_idx)
        best_alpha = torch.where(take, torch.full_like(best_alpha, float(alpha)), best_alpha)
    safe = best_idx.clamp_min(0)
    gather_idx = safe[:, None, :, :]
    selected = torch.gather(residuals, 1, gather_idx).squeeze(1)
    use = (best_idx >= 0).to(base.dtype)
    final = base + use * best_alpha * selected
    cell_gain = (base_err - best_err) * valid.bool()
    return final, best_idx, best_alpha, cell_gain


def oracle_node_batch(base, y, valid, residuals, alphas):
    """Choose one support + alpha independently for each node, shared across all horizons."""
    B, P, N = base.shape
    verr = valid.bool()
    base_abs = torch.abs(base - y)
    den = verr.sum(1).clamp_min(1)
    base_obj = (base_abs * verr).sum(1) / den
    best = base_obj.clone()
    best_idx = torch.full((B, N), -1, dtype=torch.long, device=base.device)
    best_alpha = torch.zeros((B, N), dtype=base.dtype, device=base.device)
    for alpha in alphas:
        err = torch.abs(base[:, None] + float(alpha) * residuals - y[:, None])
        vv = verr[:, None]
        obj = (err * vv).sum(2) / vv.sum(2).clamp_min(1)  # [B,K,N]
        val, idx = obj.min(1)
        take = val < best
        best = torch.where(take, val, best)
        best_idx = torch.where(take, idx, best_idx)
        best_alpha = torch.where(take, torch.full_like(best_alpha, float(alpha)), best_alpha)
    safe = best_idx.clamp_min(0)
    # Select one candidate per node, then retain all horizons from that candidate.
    rr = residuals.permute(0, 3, 1, 2)  # [B,N,K,P]
    gather = safe[:, :, None, None].expand(-1, -1, 1, P)
    selected = torch.gather(rr, 2, gather).squeeze(2).permute(0, 2, 1)
    use = (best_idx >= 0).to(base.dtype)
    final = base + selected * best_alpha[:, None, :] * use[:, None, :]
    return final, best_idx, best_alpha, base_obj - best


def oracle_region_horizon_batch(base, y, valid, direct, support, residuals, alphas, regions, weights):
    """Choose support + alpha independently for graph-region × horizon roles."""
    B, P, N = base.shape
    regions = torch.as_tensor(regions, device=base.device, dtype=torch.long)
    R = int(regions.max().item()) + 1
    final = base.clone()
    best_idx = torch.full((B, P, R), -1, dtype=torch.long, device=base.device)
    best_alpha = torch.zeros((B, P, R), dtype=base.dtype, device=base.device)
    gains = torch.zeros((B, P, R), dtype=base.dtype, device=base.device)
    # Node weights align selection with the paper's Overall/Masked/Support objective.
    node_w = (
        float(weights['overall'])
        + float(weights['masked']) * direct.float()
        + float(weights['support']) * support.float()
    )
    for r in range(R):
        nm = (regions == r)[None, None, :] & valid.bool()
        ww = node_w[:, None, :] * nm.float()
        den = ww.sum(-1).clamp_min(1e-6)  # [B,P]
        base_obj = (torch.abs(base - y) * ww).sum(-1) / den
        best = base_obj.clone()
        bi = torch.full((B, P), -1, dtype=torch.long, device=base.device)
        ba = torch.zeros((B, P), dtype=base.dtype, device=base.device)
        for alpha in alphas:
            err = torch.abs(base[:, None] + float(alpha) * residuals - y[:, None])
            obj = (err * ww[:, None]).sum(-1) / den[:, None]
            val, idx = obj.min(1)
            take = val < best
            best = torch.where(take, val, best)
            bi = torch.where(take, idx, bi)
            ba = torch.where(take, torch.full_like(ba, float(alpha)), ba)
        safe = bi.clamp_min(0)
        gather = safe[:, None, :, None].expand(-1, 1, -1, N)
        selected = torch.gather(residuals, 1, gather).squeeze(1)
        use = (bi >= 0).to(base.dtype)
        rmask = (regions == r)[None, None, :].to(base.dtype)
        final = final + rmask * use[:, :, None] * ba[:, :, None] * selected
        best_idx[:, :, r] = bi
        best_alpha[:, :, r] = ba
        gains[:, :, r] = base_obj - best
    return final, best_idx, best_alpha, gains
