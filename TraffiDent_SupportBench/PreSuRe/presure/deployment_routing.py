from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn


def need_thresholds_by_horizon(need_prob, valid_mask, keep_fraction):
    """Train-only horizon-specific thresholds for predicted Need-q90.

    need_prob / valid_mask: [B,P,N].  Thresholds are computed only from the
    supplied split and can then be frozen for dev/validation deployment.
    """
    s = np.asarray(need_prob, np.float64)
    v = np.asarray(valid_mask, bool)
    if s.shape != v.shape or s.ndim != 3:
        raise ValueError(f'need/valid shape mismatch: {s.shape} vs {v.shape}')
    f = float(keep_fraction)
    out = np.full((s.shape[1],), np.inf, np.float32)
    for h in range(s.shape[1]):
        x = s[:, h, :]
        m = v[:, h, :] & np.isfinite(x)
        if not np.any(m):
            continue
        if f >= 1.0:
            out[h] = -np.inf
        elif f <= 0.0:
            out[h] = np.inf
        else:
            out[h] = np.quantile(x[m], 1.0 - f)
    return out


def predicted_need_mask(need_prob, thresholds, valid_mask=None):
    p = np.asarray(need_prob, np.float32)
    th = np.asarray(thresholds, np.float32).reshape(1, -1, 1)
    if p.ndim != 3 or p.shape[1] != th.shape[1]:
        raise ValueError(f'need/threshold shape mismatch: {p.shape} vs {th.shape}')
    m = p >= th
    if valid_mask is not None:
        m &= np.asarray(valid_mask, bool)
    return m


def deployment_specialist_loss(pred, target, candidate_mask, valid, deployment_mask, cfg):
    """Pairwise + clear beneficial/harmful loss on a deployable Need subset.

    pred/target [B,N,L,P], candidate_mask [B,N,L], valid/deployment_mask [B,P,N].
    No current-query future is an input.  `target` is Train-only supervision.
    """
    c = cfg['deployment_aligned_utility_router_probe']['specialist_training']
    vm = valid.bool().permute(0, 2, 1)
    dm = deployment_mask.bool().permute(0, 2, 1)
    cell = vm & dm
    use = candidate_mask[:, :, :, None].expand_as(pred) & cell[:, :, None, :]
    if not torch.any(use):
        z = pred.sum() * 0.0
        return z, z, z, torch.tensor(0.0, device=pred.device), 0

    pos_thr = float(c['clear_beneficial_threshold'])
    neg_thr = float(c['clear_harm_threshold'])
    clear = use & ((target >= pos_thr) | (target <= -neg_thr))
    if torch.any(clear):
        label = (target >= pos_thr).float()
        raw = torch.nn.functional.binary_cross_entropy_with_logits(
            pred / float(c['classification_temperature']), label, reduction='none'
        )
        cls = raw[clear].mean()
    else:
        cls = pred.sum() * 0.0

    hi = target.masked_fill(~use, -1e6).argmax(2)
    lo = target.masked_fill(~use, 1e6).argmin(2)
    sp = torch.gather(pred, 2, hi[:, :, None, :]).squeeze(2)
    sn = torch.gather(pred, 2, lo[:, :, None, :]).squeeze(2)
    tp = torch.gather(target, 2, hi[:, :, None, :]).squeeze(2)
    tn = torch.gather(target, 2, lo[:, :, None, :]).squeeze(2)
    enough = (use.sum(2) >= 2) & ((tp - tn) >= float(c['rank_min_gap'])) & cell
    if torch.any(enough):
        raw_rank = torch.relu(float(c['rank_margin']) - sp + sn)
        rank = raw_rank[enough].mean()
        racc = (sp[enough] > sn[enough]).float().mean()
        rank_n = int(enough.sum().detach().cpu())
    else:
        rank = pred.sum() * 0.0
        racc = torch.tensor(0.0, device=pred.device)
        rank_n = 0

    loss = float(c['rank_weight']) * rank + float(c['classification_weight']) * cls
    return loss, rank, cls, racc, rank_n


class ExpertPreferenceRouter(nn.Module):
    """Choose which support-utility expert to trust for each node x horizon.

    Inputs are deployment-safe confidence summaries from expert scores, predicted
    Need-q90, M3 confidence, and candidate availability.  It never receives the
    current-query future or true q90 state.
    """
    def __init__(self, n_experts=4, hidden=96, dropout=0.1):
        super().__init__()
        self.n_experts = int(n_experts)
        # For each expert: max, second, margin, mean, std, top1-index -> 6E.
        # Shared: predicted Need, M3 max/second/margin/mean, candidate fraction -> 6.
        self.input_dim = 6 * self.n_experts + 6
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, int(hidden)), nn.GELU(), nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(hidden)), nn.GELU(),
            nn.Linear(int(hidden), self.n_experts),
        )

    def forward(self, x):
        return self.net(x)


def _expert_summary(score, candidate_mask, temperature):
    """score [B,N,L,P] -> summary [B,N,P,6], top1 [B,N,P]."""
    p = torch.sigmoid(score / float(temperature))
    cm = candidate_mask[:, :, :, None].expand_as(p)
    pm = p.masked_fill(~cm, -1.0)
    vals, idx = torch.sort(pm, dim=2, descending=True)
    top1 = vals[:, :, 0, :].clamp_min(0.0)
    top2 = vals[:, :, 1, :].clamp_min(0.0) if vals.shape[2] > 1 else torch.zeros_like(top1)
    count = cm.sum(2).clamp_min(1)
    clean = p.masked_fill(~cm, 0.0)
    mean = clean.sum(2) / count
    var = ((clean - mean[:, :, None, :]) ** 2 * cm.float()).sum(2) / count
    std = torch.sqrt(var.clamp_min(0.0) + 1e-8)
    denom = max(int(score.shape[2]) - 1, 1)
    top_idx = idx[:, :, 0, :].float() / float(denom)
    feat = torch.stack([top1, top2, top1 - top2, mean, std, top_idx], dim=-1)
    return feat, idx[:, :, 0, :]


def router_features(expert_scores, expert_order, need_prob, m3_conf, candidate_mask, temperature):
    """Build [B,N,P,F] deployment-safe router features."""
    feats = []
    top_ids = []
    for name in expert_order:
        f, ti = _expert_summary(expert_scores[name], candidate_mask, temperature)
        feats.append(f)
        top_ids.append(ti)

    need = need_prob.float().permute(0, 2, 1)  # [B,N,P]
    conf = m3_conf.float()
    cm = candidate_mask.bool()
    cmf = cm.float()
    cnt = cmf.sum(2).clamp_min(1.0)
    csort, _ = torch.sort(conf.masked_fill(~cm, -1e6), dim=2, descending=True)
    c1 = csort[:, :, 0].clamp_min(0.0)
    c2 = csort[:, :, 1].clamp_min(0.0) if csort.shape[2] > 1 else torch.zeros_like(c1)
    cmean = (conf.masked_fill(~cm, 0.0).sum(2) / cnt).clamp_min(0.0)
    cfrac = cnt / float(max(conf.shape[2], 1))
    P = need.shape[-1]
    shared = torch.stack([
        need,
        c1[:, :, None].expand(-1, -1, P),
        c2[:, :, None].expand(-1, -1, P),
        (c1 - c2)[:, :, None].expand(-1, -1, P),
        cmean[:, :, None].expand(-1, -1, P),
        cfrac[:, :, None].expand(-1, -1, P),
    ], dim=-1)
    return torch.cat(feats + [shared], dim=-1), top_ids


def expert_selected_utility(score, target, candidate_mask):
    cm = candidate_mask[:, :, :, None].expand_as(score)
    idx = score.masked_fill(~cm, -1e6).argmax(2)
    util = torch.gather(target, 2, idx[:, :, None, :]).squeeze(2)
    return util, idx


def router_targets(expert_scores, expert_order, target, candidate_mask, valid, min_gap):
    """Privileged Train-only expert-preference labels.

    Label is the expert whose own selected support has the highest true utility.
    Ambiguous cells (best-vs-second gap < min_gap) are excluded.
    """
    utils = []
    for name in expert_order:
        u, _ = expert_selected_utility(expert_scores[name], target, candidate_mask)
        utils.append(u)
    u = torch.stack(utils, dim=-1)  # [B,N,P,E]
    vals, idx = torch.sort(u, dim=-1, descending=True)
    gap = vals[..., 0] - vals[..., 1]
    avail = candidate_mask.any(2)[:, :, None].expand_as(gap)
    vm = valid.bool().permute(0, 2, 1)
    use = vm & avail & (gap >= float(min_gap))
    return idx[..., 0], use, u


def router_loss(logits, labels, use, need_prob, need_weight_lambda):
    if not torch.any(use):
        z = logits.sum() * 0.0
        return z, torch.tensor(0.0, device=logits.device), 0
    raw = torch.nn.functional.cross_entropy(logits[use], labels[use], reduction='none')
    need = need_prob.float().permute(0, 2, 1)
    w = 1.0 + float(need_weight_lambda) * need
    loss = (raw * w[use]).sum() / w[use].sum().clamp_min(1.0)
    acc = (logits.argmax(-1)[use] == labels[use]).float().mean()
    return loss, acc, int(use.sum().detach().cpu())


def route_candidate_scores(router_logits, expert_scores, expert_order, temperature, soft=False):
    """Return candidate scores [B,N,L,P] on a common probability scale."""
    probs = torch.stack([
        torch.sigmoid(expert_scores[name] / float(temperature)) for name in expert_order
    ], dim=-1)  # [B,N,L,P,E]
    if soft:
        w = torch.softmax(router_logits, dim=-1)[:, :, None, :, :]
        return (probs * w).sum(-1)
    choice = router_logits.argmax(-1)  # [B,N,P]
    oh = torch.nn.functional.one_hot(choice, num_classes=len(expert_order)).float()[:, :, None, :, :]
    return (probs * oh).sum(-1)


def oracle_expert_scores(expert_scores, expert_order, target, candidate_mask, temperature):
    """Privileged ceiling: choose the expert whose own selected support is best."""
    _, _, u = router_targets(expert_scores, expert_order, target, candidate_mask,
                             torch.ones(target.shape[0], target.shape[-1], target.shape[1],
                                        dtype=torch.bool, device=target.device), -1e9)
    choice = u.argmax(-1)  # [B,N,P]
    probs = torch.stack([
        torch.sigmoid(expert_scores[name] / float(temperature)) for name in expert_order
    ], dim=-1)
    oh = torch.nn.functional.one_hot(choice, num_classes=len(expert_order)).float()[:, :, None, :, :]
    return (probs * oh).sum(-1)


@dataclass
class RouterAccuracyAccumulator:
    correct: int = 0
    total: int = 0

    def update(self, logits, labels, use):
        if torch.any(use):
            self.correct += int(((logits.argmax(-1) == labels) & use).sum().detach().cpu())
            self.total += int(use.sum().detach().cpu())

    def summary(self):
        return {'expert_preference_accuracy': None if self.total == 0 else self.correct / self.total,
                'expert_preference_count': self.total}
