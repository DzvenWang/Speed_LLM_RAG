"""Training objective of ReShift-STLA v0.2 (labels are Train/Val supervision only)."""
from __future__ import annotations
import torch
from torch.nn import functional as F


PART_KEYS = ('mae', 'rmse', 'base_loss', 'rank', 'risk', 'cf_distill', 'alignment', 'evolution_rank', 'evolution_risk',
             'replay_rank', 'replay_risk', 'gate_evolution', 'gate_replay', 'gate_cf', 'loss')


def _mmean(x, m):
    """Masked mean without boolean indexing (no host synchronisation)."""
    m = m.to(x.dtype); return (x * m).sum() / m.sum().clamp_min(1.0)


def utility_rank_risk(branch, base_n, y_n, valid, cfg):
    """Candidate utility ranking against the None option, and the risk-gate target.

    utility_c = |B - Y| - |B + r_c - Y|  (> 0: candidate c would help this element)
    """
    tc = cfg['training']; margin = float(tc.get('utility_margin', 0.02)); sm = float(tc.get('score_margin', 0.0))
    r = branch['residuals'].detach().float()
    util = (base_n[:, None] - y_n[:, None]).abs() - (base_n[:, None] + r - y_n[:, None]).abs()
    avail = branch['available'].bool() & valid[:, None]
    pos = ((util > margin) & avail).float(); neg = ((util < -margin) & avail).float()
    logits = branch['logits'].float(); rel = logits[:, :-1] - logits[:, -1:]
    count = (pos.sum() + neg.sum()).clamp_min(1)
    rank = ((F.softplus(sm - rel) * pos).sum() + (F.softplus(sm + rel) * neg).sum()) / count
    proposal = base_n + branch['ungated'].detach().float()
    gain = (base_n - y_n).abs() - (proposal - y_n).abs()
    bce = F.binary_cross_entropy_with_logits(branch['risk_logit'].float(), (gain > margin).float(), reduction='none')
    return rank, _mmean(bce, valid)


def compute_losses(out, b, mean, std, cfg, alignment=None):
    """Returns (total loss, parts) where parts is a detached [len(PART_KEYS)] tensor.

    Nothing here forces a GPU->host copy; callers convert `parts` once per epoch.
    """
    tc = cfg['training']
    valid = b['future_valid'].bool(); y = b['y_raw'].float(); yn = b['y_norm'].float()
    vf = valid.float(); p = out['prediction'].float() * std + mean
    err = torch.where(valid, p - y, torch.zeros_like(p))
    main = _mmean(err.abs(), vf); rmse = torch.sqrt(_mmean(err.square(), vf) + 1e-8)
    base_n = out['base_prediction'].detach().float()
    berr = torch.where(valid, out['base_prediction'].float() * std + mean - y, torch.zeros_like(p))
    base_loss = _mmean(berr.abs(), vf)
    zero = main.new_zeros(()); extra = {}
    rank = risk = zero
    for name in ('evolution', 'replay'):
        if out.get(name) is not None:
            r1, r2 = utility_rank_risk(out[name], base_n, yn, valid, cfg)
            rank = rank + r1; risk = risk + r2; extra[f'{name}_rank'] = r1; extra[f'{name}_risk'] = r2
    cf = zero
    if 'cf_target' in b and out.get('cf_shift') is not None and float(tc.get('cf_weight', 0)) > 0:
        tv = b['cf_target_valid'].bool() & valid
        hub = F.smooth_l1_loss(out['cf_shift'].float(), torch.nan_to_num(b['cf_target'].float()),
                               beta=float(tc.get('cf_huber_beta', 0.1)), reduction='none')
        cf = _mmean(hub, tv)
    align = zero if alignment is None else alignment
    total = (main + float(tc.get('rmse_weight', 0)) * rmse + float(tc.get('rank_weight', 0)) * rank
             + float(tc.get('risk_weight', 0)) * risk + float(tc.get('base_preserve_weight', 0)) * base_loss
             + float(tc.get('cf_weight', 0)) * cf + float(tc.get('alignment_weight', 0)) * align)
    g = out['gates'].detach().float().mean((0, 1, 2))
    vals = {'mae': main, 'rmse': rmse, 'base_loss': base_loss, 'rank': rank, 'risk': risk, 'cf_distill': cf,
            'alignment': align, **extra, 'gate_evolution': g[0], 'gate_replay': g[1], 'gate_cf': g[2], 'loss': total}
    parts = torch.stack([vals.get(k, zero).detach().float().reshape(()) for k in PART_KEYS])
    return total, parts


def parts_dict(parts):
    """Convert an accumulated parts tensor to a {name: float} dict (one host copy)."""
    return dict(zip(PART_KEYS, parts.detach().cpu().tolist()))


def alignment_loss(model, b, repo, device, temperature=.1, out=None):
    """Train-only InfoNCE between an incomplete query and its own complete window.

    Pairs are identified by the exact historical window_start; temporally
    overlapping windows are never used as negatives.
    """
    if 'repo_row' in b:
        # Train queries: memory row of the same complete window, looked up on the GPU.
        mids = b['repo_row'].long(); ok = mids >= 0
        rows = torch.arange(len(mids), device=mids.device); mids = mids.clamp_min(0)
    else:
        if not hasattr(repo, 'row_by_start'): repo.row_by_start = {int(t): i for i, t in enumerate(repo.starts)}
        starts = b['window_start'].detach().cpu().tolist()
        paired = [(j, repo.row_by_start[int(w)]) for j, w in enumerate(starts) if int(w) in repo.row_by_start]
        if len(paired) < 2: return torch.zeros((), device=device)
        rows = torch.tensor([p[0] for p in paired], device=device, dtype=torch.long)
        mids = torch.tensor([p[1] for p in paired], device=device, dtype=torch.long); ok = torch.ones_like(rows, dtype=torch.bool)
    mx, mm, mt, md = repo.fetch(mids, device)
    with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        mem = model.memory_encoder(mx.float(), mm.float(), mt, md)
    q = {'temporal_key': out['query_temporal_keys'][rows], 'spatial_key': out['query_spatial_keys'][rows]}
    qs = b['window_start'][rows].long(); overlap = (qs[:, None] - qs[None, :]).abs() < model.H + model.P
    diag = torch.eye(len(rows), device=device, dtype=torch.bool); loss = torch.zeros((), device=device)
    with torch.autocast(device_type=torch.device(device).type, enabled=False):
        for key in ('temporal_key', 'spatial_key'):
            a = F.normalize(q[key].float(), dim=-1); m = F.normalize(mem[key].float().detach(), dim=-1)
            logits = ((a @ m.T) / max(float(temperature), 1e-5)).masked_fill((overlap & ~diag) | ~ok[None, :], -1e4)
            ce = F.cross_entropy(logits, torch.arange(len(rows), device=device), reduction='none')
            loss = loss + _mmean(ce, ok)
    return loss / 2.
