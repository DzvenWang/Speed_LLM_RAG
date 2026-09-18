from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F

from .probes import binary_auroc


def strong_harm_focal_loss(logit, target, candidate_mask, valid_cell, threshold=0.02,
                           gamma=2.0, alpha=0.5, rank_score=None, top_rank_weight=1.0,
                           top_rank_k=4):
    """Independent strong-harm objective for v0.9A.2.

    The frozen v0.9A ranker supplies ``rank_score``.  The veto learns only
    y=1[target <= -threshold].  Optional top-rank weighting focuses learning on
    candidates that the deployment ranker is actually likely to inspect.
    """
    use = candidate_mask[:, :, :, None] & valid_cell[:, :, None, :]
    if not torch.any(use):
        z = logit.sum() * 0.0
        return z, {'strong_harm_rate': 0.0, 'sample_n': 0}
    y = (target <= -float(threshold)).float()
    p = torch.sigmoid(logit)
    pt = torch.where(y > 0.5, p, 1.0 - p).clamp(1e-6, 1.0 - 1e-6)
    a = torch.where(y > 0.5, torch.full_like(y, float(alpha)), torch.full_like(y, 1.0 - float(alpha)))
    loss = -a * torch.pow(1.0 - pt, float(gamma)) * torch.log(pt)

    if rank_score is not None and float(top_rank_weight) > 1.0 and int(top_rank_k) > 0:
        rs = rank_score.masked_fill(~use, -1e9)
        K = rs.shape[2]
        kk = min(int(top_rank_k), K)
        top = torch.topk(rs, kk, dim=2).indices
        topmask = torch.zeros_like(use)
        topmask.scatter_(2, top, True)
        w = torch.where(topmask, torch.full_like(loss, float(top_rank_weight)), torch.ones_like(loss))
        loss = loss * w

    return loss[use].mean(), {
        'strong_harm_rate': float(y[use].mean().detach().cpu()),
        'sample_n': int(use.sum().detach().cpu()),
    }


def frozen_rank_order(rank_score, candidate_mask, valid_cell):
    """Return frozen v0.9A candidate order for each node-horizon cell.

    This helper is the single source of truth for policy calibration and final
    evaluation.  It never uses veto probabilities, so the safety head cannot
    reorder candidates.
    """
    use = candidate_mask[:, :, :, None] & valid_cell[:, :, None, :]
    rs = rank_score.masked_fill(~use, -1e9)
    return torch.argsort(rs, dim=2, descending=True)


def sequential_veto_choice(rank_score, harm_prob, candidate_mask, valid_cell,
                           veto_threshold, max_checks, order=None):
    """Rank-then-veto policy without reordering safe candidates.

    Returns selected candidate index [B,N,P], acceptance mask [B,N,P], number of
    inspected candidates [B,N,P], and selected frozen rank score [B,N,P].
    """
    use = candidate_mask[:, :, :, None] & valid_cell[:, :, None, :]
    rs = rank_score.masked_fill(~use, -1e9)
    if order is None:
        order = frozen_rank_order(rank_score, candidate_mask, valid_cell)
    B, N, K, P = rs.shape
    m = min(max(int(max_checks), 1), K)
    chosen = torch.zeros((B, N, P), dtype=torch.long, device=rs.device)
    accepted = torch.zeros((B, N, P), dtype=torch.bool, device=rs.device)
    checks = torch.zeros((B, N, P), dtype=torch.int16, device=rs.device)
    selected_score = torch.full((B, N, P), -1e9, dtype=rs.dtype, device=rs.device)

    for j in range(m):
        idx = order[:, :, j, :]
        hp = torch.gather(harm_prob, 2, idx[:, :, None, :]).squeeze(2)
        sc = torch.gather(rs, 2, idx[:, :, None, :]).squeeze(2)
        available = valid_cell & (~accepted) & (sc > -1e8)
        checks = torch.where(available, torch.full_like(checks, j + 1), checks)
        safe = available & (hp < float(veto_threshold))
        chosen = torch.where(safe, idx, chosen)
        selected_score = torch.where(safe, sc, selected_score)
        accepted |= safe
    return chosen, accepted, checks, selected_score


@dataclass
class HarmClassifierAccumulator:
    sample_cap_per_update: int = 8192
    seed: int = 10
    score_samples: list = field(default_factory=list)
    label_samples: list = field(default_factory=list)
    updates: int = 0

    def update(self, harm_logit, target, candidate_mask, valid_cell, threshold):
        use = candidate_mask[:, :, :, None] & valid_cell[:, :, None, :]
        if not torch.any(use):
            return
        s = torch.sigmoid(harm_logit[use]).detach().cpu().numpy()
        y = (target[use] <= -float(threshold)).detach().cpu().numpy().astype(np.int8)
        take = min(len(s), int(self.sample_cap_per_update))
        if take:
            rng = np.random.default_rng(self.seed + 991 * self.updates)
            ids = rng.choice(len(s), size=take, replace=False) if len(s) > take else np.arange(len(s))
            self.score_samples.append(s[ids].astype(np.float32, copy=False))
            self.label_samples.append(y[ids])
        self.updates += 1

    def summary(self, threshold=0.5):
        if not self.score_samples:
            return {'auroc': 0.5, 'precision': 0.0, 'recall': 0.0, 'positive_rate': 0.0, 'sample_n': 0}
        s = np.concatenate(self.score_samples)
        y = np.concatenate(self.label_samples).astype(bool)
        pred = s >= float(threshold)
        tp = int(np.sum(pred & y)); fp = int(np.sum(pred & ~y)); fn = int(np.sum(~pred & y))
        auc = binary_auroc(s, y.astype(np.int64))
        return {
            'auroc': 0.5 if auc is None else float(auc),
            'precision': float(tp / max(tp + fp, 1)),
            'recall': float(tp / max(tp + fn, 1)),
            'positive_rate': float(np.mean(y)),
            'predicted_positive_rate': float(np.mean(pred)),
            'sample_n': int(len(s)),
        }


@dataclass
class RankThenVetoAccumulator:
    beneficial_threshold: float
    harmful_threshold: float
    total_n: int = 0
    accepted_n: int = 0
    accepted_beneficial: int = 0
    accepted_harmful: int = 0
    accepted_strong_harm: int = 0
    oracle_beneficial: int = 0
    policy_utility_sum: float = 0.0
    accepted_utility_sum: float = 0.0
    safe_oracle_utility_sum: float = 0.0
    safe_regret_sum: float = 0.0
    checks_sum: float = 0.0
    fallback_n: int = 0

    def update(self, chosen_idx, accepted, checks, target, candidate_mask, valid_cell):
        use = candidate_mask[:, :, :, None] & valid_cell[:, :, None, :]
        mt = target.masked_fill(~use, -1e9)
        best = mt.max(2).values
        chosen_u = torch.gather(target, 2, chosen_idx[:, :, None, :]).squeeze(2)
        ok = (use.sum(2) > 0) & valid_cell
        if not torch.any(ok):
            return
        av = accepted & ok
        cv = chosen_u[ok]
        bv = best[ok]
        aok = av[ok]
        policy = torch.where(aok, cv, torch.zeros_like(cv))
        oracle_safe = torch.clamp(bv, min=0.0)
        self.total_n += int(cv.numel())
        self.accepted_n += int(aok.sum().detach().cpu())
        if torch.any(aok):
            au = cv[aok]
            self.accepted_beneficial += int((au >= float(self.beneficial_threshold)).sum().detach().cpu())
            self.accepted_harmful += int((au < 0.0).sum().detach().cpu())
            self.accepted_strong_harm += int((au <= -float(self.harmful_threshold)).sum().detach().cpu())
            self.accepted_utility_sum += float(au.sum().detach().cpu())
        self.oracle_beneficial += int((bv >= float(self.beneficial_threshold)).sum().detach().cpu())
        self.policy_utility_sum += float(policy.sum().detach().cpu())
        self.safe_oracle_utility_sum += float(oracle_safe.sum().detach().cpu())
        self.safe_regret_sum += float((oracle_safe - policy).sum().detach().cpu())
        self.checks_sum += float(checks[ok].float().sum().detach().cpu())
        self.fallback_n += int(((checks > 1) & av & ok).sum().detach().cpu())

    def summary(self):
        return {
            'cell_count': int(self.total_n),
            'coverage': float(self.accepted_n / max(self.total_n, 1)),
            'accepted_count': int(self.accepted_n),
            'accepted_beneficial_rate': float(self.accepted_beneficial / max(self.accepted_n, 1)),
            'accepted_harmful_rate': float(self.accepted_harmful / max(self.accepted_n, 1)),
            'accepted_strong_harm_rate': float(self.accepted_strong_harm / max(self.accepted_n, 1)),
            'beneficial_capture_ratio': float(self.accepted_beneficial / max(self.oracle_beneficial, 1)),
            'mean_policy_relative_utility': float(self.policy_utility_sum / max(self.total_n, 1)),
            'mean_accepted_relative_utility': float(self.accepted_utility_sum / max(self.accepted_n, 1)),
            'mean_safe_oracle_relative_utility': float(self.safe_oracle_utility_sum / max(self.total_n, 1)),
            'mean_safe_policy_regret': float(self.safe_regret_sum / max(self.total_n, 1)),
            'mean_checks': float(self.checks_sum / max(self.total_n, 1)),
            'fallback_rate': float(self.fallback_n / max(self.accepted_n, 1)),
        }
