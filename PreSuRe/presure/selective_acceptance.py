from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np
import torch
import torch.nn.functional as F

from .probes import binary_auroc


def _gather_candidate(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather candidate dimension=2 using idx [B,N,P]."""
    if x.ndim == 4:  # [B,N,K,P]
        return torch.gather(x, 2, idx[:, :, None, :]).squeeze(2)
    if x.ndim == 5:  # [B,N,K,P,D]
        D = x.shape[-1]
        gi = idx[:, :, None, :, None].expand(-1, -1, 1, -1, D)
        return torch.gather(x, 2, gi).squeeze(2)
    raise ValueError(f'Unsupported candidate tensor rank: {x.ndim}')


def frozen_top1_acceptance_context(role_features, mu, sigma, beneficial_logit, rank_score,
                                   candidate_mask, valid_cell):
    """Build deployment-safe features for the *already selected* frozen-v0.9A Top-1.

    The acceptance model never sees alternative-candidate future utility and never
    reorders candidates.  It receives the frozen Top-1 role representation plus
    rank-context statistics that describe how decisive the Top-1 ranking is.

    Returns
    -------
    selected_idx : [B,N,P]
    features     : [B,N,P,D_role+11]
    selected_valid: [B,N,P]
    """
    use = candidate_mask[:, :, :, None] & valid_cell[:, :, None, :]
    masked = rank_score.masked_fill(~use, -1e9)
    B, N, K, P = masked.shape
    kk = min(3, K)
    vals, ids = torch.topk(masked, kk, dim=2)
    idx1 = ids[:, :, 0, :]
    s1 = vals[:, :, 0, :]
    s2 = vals[:, :, 1, :] if kk >= 2 else s1
    s3 = vals[:, :, 2, :] if kk >= 3 else s2

    r1 = _gather_candidate(role_features, idx1)
    mu1 = _gather_candidate(mu, idx1)
    sg1 = _gather_candidate(sigma, idx1)
    bl1 = _gather_candidate(beneficial_logit, idx1)

    uf = use.float()
    count = uf.sum(2).clamp_min(1.0)
    safe_score = torch.where(use, rank_score, torch.zeros_like(rank_score))
    mean = safe_score.sum(2) / count
    var = (torch.where(use, (rank_score - mean[:, :, None, :]) ** 2, torch.zeros_like(rank_score)).sum(2) / count)
    std = torch.sqrt(var.clamp_min(1e-8))
    z1 = (s1 - mean) / std.clamp_min(1e-4)

    # Normalized entropy of frozen candidate scores: high => ambiguous ranking.
    probs = torch.softmax(masked, dim=2)
    ent = -(probs * torch.log(probs.clamp_min(1e-8))).sum(2)
    ent_denom = torch.log(count.clamp_min(2.0))
    ent = ent / ent_denom.clamp_min(1e-6)
    ent = torch.where(count > 1.0, ent, torch.zeros_like(ent))
    count_frac = count / float(K)
    rel_unc = sg1 / (torch.abs(mu1) + 0.05)

    scalars = torch.stack([
        mu1, sg1, bl1, s1,
        s1 - s2, s1 - s3,
        mean, std, z1, ent, count_frac, rel_unc,
    ], dim=-1)
    features = torch.cat([r1, scalars], dim=-1)
    selected_valid = valid_cell & (use.sum(2) > 0) & (s1 > -1e8)
    return idx1, features, selected_valid


def selected_target(target: torch.Tensor, selected_idx: torch.Tensor) -> torch.Tensor:
    return _gather_candidate(target, selected_idx)


def selective_acceptance_loss(logit, selected_utility, valid_cell, cfg,
                              masked_cell=None, support_cell=None):
    """Utility-aware binary loss on frozen Top-1 supports only.

    Label is positive iff the selected support has strictly positive utility.  A
    narrow near-zero band receives reduced weight because its sign is intrinsically
    unstable, while larger-magnitude utility receives more weight.  This makes the
    objective align with USE-vs-ABSTAIN rather than candidate ranking.
    """
    c = cfg['predictive_utility_selective_acceptance']['training']
    use = valid_cell
    if not torch.any(use):
        z = logit.sum() * 0.0
        return z, {'positive_rate': 0.0, 'ambiguous_rate': 0.0, 'sample_n': 0}
    y = (selected_utility > float(c['positive_utility_threshold'])).float()
    per = F.binary_cross_entropy_with_logits(logit, y, reduction='none')

    absu = torch.abs(selected_utility)
    w = 1.0 + float(c['utility_weight_scale']) * absu
    w = torch.clamp(w, max=float(c['utility_weight_clip']))
    amb = absu < float(c['ambiguous_abs_utility'])
    w = torch.where(amb, w * float(c['ambiguous_weight']), w)
    if masked_cell is not None:
        w = torch.where(masked_cell, w * float(c['masked_weight']), w)
    if support_cell is not None:
        w = torch.where(support_cell, w * float(c['support_weight']), w)

    loss = (per[use] * w[use]).sum() / w[use].sum().clamp_min(1e-6)
    return loss, {
        'positive_rate': float(y[use].mean().detach().cpu()),
        'ambiguous_rate': float(amb[use].float().mean().detach().cpu()),
        'sample_n': int(use.sum().detach().cpu()),
    }


@dataclass
class AcceptanceClassifierAccumulator:
    positive_utility_threshold: float = 0.0
    sample_cap_per_update: int = 200000
    seed: int = 10
    scores: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    utilities: list = field(default_factory=list)
    updates: int = 0

    def update(self, logit, selected_utility, valid_cell):
        if not torch.any(valid_cell):
            return
        p = torch.sigmoid(logit[valid_cell]).detach().cpu().numpy().astype(np.float32, copy=False)
        u = selected_utility[valid_cell].detach().cpu().numpy().astype(np.float32, copy=False)
        y = (u > float(self.positive_utility_threshold)).astype(np.int8)
        take = min(len(p), int(self.sample_cap_per_update))
        if take:
            rng = np.random.default_rng(self.seed + 997 * self.updates)
            ids = rng.choice(len(p), size=take, replace=False) if len(p) > take else np.arange(len(p))
            self.scores.append(p[ids]); self.labels.append(y[ids]); self.utilities.append(u[ids])
        self.updates += 1

    def summary(self, threshold=0.5):
        if not self.scores:
            return {'auroc': 0.5, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                    'positive_rate': 0.0, 'predicted_positive_rate': 0.0,
                    'brier': 0.25, 'sample_n': 0}
        p = np.concatenate(self.scores)
        y = np.concatenate(self.labels).astype(bool)
        pred = p >= float(threshold)
        tp = int(np.sum(pred & y)); fp = int(np.sum(pred & ~y)); fn = int(np.sum(~pred & y))
        precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
        auc = binary_auroc(p, y.astype(np.int64))
        return {
            'auroc': 0.5 if auc is None else float(auc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(2 * precision * recall / max(precision + recall, 1e-12)),
            'positive_rate': float(np.mean(y)),
            'predicted_positive_rate': float(np.mean(pred)),
            'brier': float(np.mean((p - y.astype(np.float32)) ** 2)),
            'sample_n': int(len(p)),
        }


@dataclass
class SelectiveAcceptanceAccumulator:
    beneficial_threshold: float
    harmful_threshold: float
    total_n: int = 0
    accepted_n: int = 0
    accepted_positive: int = 0
    accepted_beneficial: int = 0
    accepted_harmful: int = 0
    accepted_strong_harm: int = 0
    oracle_beneficial: int = 0
    policy_utility_sum: float = 0.0
    accepted_utility_sum: float = 0.0
    safe_oracle_utility_sum: float = 0.0
    safe_regret_sum: float = 0.0

    def update(self, selected_utility, accept, best_utility, valid_cell):
        ok = valid_cell
        if not torch.any(ok):
            return
        uv = selected_utility[ok]
        av = accept[ok]
        bv = best_utility[ok]
        policy = torch.where(av, uv, torch.zeros_like(uv))
        oracle_safe = torch.clamp(bv, min=0.0)
        self.total_n += int(uv.numel())
        self.accepted_n += int(av.sum().detach().cpu())
        if torch.any(av):
            au = uv[av]
            self.accepted_positive += int((au > 0.0).sum().detach().cpu())
            self.accepted_beneficial += int((au >= float(self.beneficial_threshold)).sum().detach().cpu())
            self.accepted_harmful += int((au < 0.0).sum().detach().cpu())
            self.accepted_strong_harm += int((au <= -float(self.harmful_threshold)).sum().detach().cpu())
            self.accepted_utility_sum += float(au.sum().detach().cpu())
        self.oracle_beneficial += int((bv >= float(self.beneficial_threshold)).sum().detach().cpu())
        self.policy_utility_sum += float(policy.sum().detach().cpu())
        self.safe_oracle_utility_sum += float(oracle_safe.sum().detach().cpu())
        self.safe_regret_sum += float((oracle_safe - policy).sum().detach().cpu())

    def summary(self):
        return {
            'cell_count': int(self.total_n),
            'coverage': float(self.accepted_n / max(self.total_n, 1)),
            'accepted_count': int(self.accepted_n),
            'accepted_positive_utility_rate': float(self.accepted_positive / max(self.accepted_n, 1)),
            'accepted_beneficial_rate': float(self.accepted_beneficial / max(self.accepted_n, 1)),
            'accepted_harmful_rate': float(self.accepted_harmful / max(self.accepted_n, 1)),
            'accepted_strong_harm_rate': float(self.accepted_strong_harm / max(self.accepted_n, 1)),
            'beneficial_capture_ratio': float(self.accepted_beneficial / max(self.oracle_beneficial, 1)),
            'mean_policy_relative_utility': float(self.policy_utility_sum / max(self.total_n, 1)),
            'mean_accepted_relative_utility': float(self.accepted_utility_sum / max(self.accepted_n, 1)),
            'mean_safe_oracle_relative_utility': float(self.safe_oracle_utility_sum / max(self.total_n, 1)),
            'mean_safe_policy_regret': float(self.safe_regret_sum / max(self.total_n, 1)),
        }
