from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F

from .probes import binary_auroc


def normalize_coarse_scores(scores: np.ndarray) -> np.ndarray:
    """Per-query z-normalize coarse retrieval scores for stable model input."""
    s = np.asarray(scores, np.float32)
    mu = s.mean(axis=1, keepdims=True)
    sd = s.std(axis=1, keepdims=True) + 1e-5
    return np.clip((s - mu) / sd, -4.0, 4.0).astype(np.float32)


def node_horizon_utility_targets(true_need_norm, rr_norm, candidate_mask, alpha_grid,
                                 gain_floor=0.1, utility_clip=1.5):
    """Future-grounded training target for v0.9A.

    Args:
      true_need_norm: [B,P,N] normalized future residual (Y - base) / std.
      rr_norm: [B,N,K,P] historical Train residuals / std.
      candidate_mask: [B,N,K] leakage-safe candidate availability.

    Returns:
      relative_utility [B,N,K,P], best_alpha [B,N,K,P].

    Query future is used only here for Train supervision / diagnostic evaluation;
    it is never passed to the reranker forward path.
    """
    n = true_need_norm.permute(0, 2, 1)[:, :, None, :]  # [B,N,1,P]
    base_err = torch.abs(n)
    best_gain = torch.full_like(rr_norm, -1e9)
    best_alpha = torch.zeros_like(rr_norm)
    for a in alpha_grid:
        gain = base_err - torch.abs(n - float(a) * rr_norm)
        take = gain > best_gain
        best_gain = torch.where(take, gain, best_gain)
        best_alpha = torch.where(take, torch.full_like(best_alpha, float(a)), best_alpha)
    rel = best_gain / (base_err + float(gain_floor))
    rel = torch.clamp(rel, -float(utility_clip), float(utility_clip))
    cm = candidate_mask[:, :, :, None]
    rel = rel.masked_fill(~cm, -float(utility_clip))
    best_alpha = best_alpha.masked_fill(~cm, 0.0)
    return rel, best_alpha


def risk_score(mu, sigma, beta):
    return mu - float(beta) * sigma


def _clear_mask(target, beneficial_threshold, harmful_threshold):
    return (target >= float(beneficial_threshold)) | (target <= -float(harmful_threshold))


def _pairwise_best_worst(score, target, use, margin, min_gap):
    mt_hi = target.masked_fill(~use, -1e6)
    mt_lo = target.masked_fill(~use, 1e6)
    hi = mt_hi.argmax(2)
    lo = mt_lo.argmin(2)
    sp = torch.gather(score, 2, hi[:, :, None, :]).squeeze(2)
    sn = torch.gather(score, 2, lo[:, :, None, :]).squeeze(2)
    tp = torch.gather(target, 2, hi[:, :, None, :]).squeeze(2)
    tn = torch.gather(target, 2, lo[:, :, None, :]).squeeze(2)
    enough = (use.sum(2) >= 2) & ((tp - tn) >= float(min_gap))
    if not torch.any(enough):
        z = score.sum() * 0.0
        return z, torch.tensor(0.0, device=score.device), 0
    raw = torch.relu(float(margin) - sp + sn)
    loss = raw[enough].mean()
    acc = (sp[enough] > sn[enough]).float().mean()
    return loss, acc, int(enough.sum().detach().cpu())


def _hard_harmful_rank(score, target, use, coarse_score, beneficial_threshold, harmful_threshold,
                        margin, similarity_weight=1.0, utility_weight=0.5):
    """Rank the best beneficial candidate above a similar-but-harmful candidate."""
    B, N, K, P = target.shape
    positive = use & (target >= float(beneficial_threshold))
    harmful = use & (target <= -float(harmful_threshold))
    pos_target = target.masked_fill(~positive, -1e6)
    pos_idx = pos_target.argmax(2)
    # high coarse similarity + strong negative utility => hardest harmful negative
    cs = coarse_score[:, None, :, None].expand(B, N, K, P)
    hard_value = float(similarity_weight) * cs + float(utility_weight) * torch.relu(-target)
    hard_value = hard_value.masked_fill(~harmful, -1e6)
    neg_idx = hard_value.argmax(2)
    sp = torch.gather(score, 2, pos_idx[:, :, None, :]).squeeze(2)
    sn = torch.gather(score, 2, neg_idx[:, :, None, :]).squeeze(2)
    ok = positive.any(2) & harmful.any(2)
    if not torch.any(ok):
        z = score.sum() * 0.0
        return z, torch.tensor(0.0, device=score.device), 0
    raw = torch.relu(float(margin) - sp + sn)
    loss = raw[ok].mean()
    acc = (sp[ok] > sn[ok]).float().mean()
    return loss, acc, int(ok.sum().detach().cpu())


def predictive_utility_loss(mu, sigma, beneficial_logit, target, candidate_mask, valid_cell,
                            coarse_score, cfg):
    """Unified v0.9A objective: utility + ranking + benefit + hard harmful."""
    c = cfg['predictive_utility_retrieval']['training']
    use = candidate_mask[:, :, :, None] & valid_cell[:, :, None, :]
    if not torch.any(use):
        z = mu.sum() * 0.0
        return z, {'utility': z, 'rank': z, 'benefit': z, 'hard': z,
                   'rank_acc': 0.0, 'hard_acc': 0.0, 'rank_n': 0, 'hard_n': 0}

    # Heteroscedastic robust regression.  log(sigma) prevents unlimited uncertainty.
    huber = F.smooth_l1_loss(mu, target, beta=float(c['huber_beta']), reduction='none')
    reg_raw = huber / sigma.clamp_min(1e-4) + torch.log(sigma.clamp_min(1e-4))
    utility = reg_raw[use].mean()

    safe = risk_score(mu, sigma, float(c.get('train_risk_beta', 0.25)))
    rank, rank_acc, rank_n = _pairwise_best_worst(
        safe, target, use, float(c['rank_margin']), float(c['rank_min_gap'])
    )

    clear = use & _clear_mask(target, c['beneficial_threshold'], c['harmful_threshold'])
    if torch.any(clear):
        label = (target >= float(c['beneficial_threshold'])).float()
        benefit = F.binary_cross_entropy_with_logits(
            beneficial_logit[clear] / float(c['classification_temperature']), label[clear]
        )
    else:
        benefit = mu.sum() * 0.0

    hard, hard_acc, hard_n = _hard_harmful_rank(
        safe, target, use, coarse_score,
        float(c['beneficial_threshold']), float(c['harmful_threshold']),
        float(c['hard_margin']), float(c['hard_similarity_weight']), float(c['hard_utility_weight'])
    )

    loss = (
        float(c['utility_weight']) * utility
        + float(c['rank_weight']) * rank
        + float(c['classification_weight']) * benefit
        + float(c['hard_harmful_weight']) * hard
    )
    return loss, {
        'utility': utility, 'rank': rank, 'benefit': benefit, 'hard': hard,
        'rank_acc': float(rank_acc.detach().cpu()), 'hard_acc': float(hard_acc.detach().cpu()),
        'rank_n': rank_n, 'hard_n': hard_n,
    }


@dataclass
class UtilitySelectionAccumulator:
    beneficial_threshold: float
    harmful_threshold: float
    sample_cap_per_update: int = 4096
    seed: int = 10
    selected_n: int = 0
    selected_beneficial: int = 0
    selected_harmful: int = 0
    selected_strong_harm: int = 0
    oracle_beneficial: int = 0
    pair_ok: int = 0
    pair_n: int = 0
    top1_ok: int = 0
    top1_n: int = 0
    selected_utility_sum: float = 0.0
    regret_sum: float = 0.0
    score_samples: list = field(default_factory=list)
    target_samples: list = field(default_factory=list)
    updates: int = 0

    def update(self, score, target, candidate_mask, valid_cell):
        use = candidate_mask[:, :, :, None] & valid_cell[:, :, None, :]
        ms = score.masked_fill(~use, -1e6)
        mt = target.masked_fill(~use, -1e6)
        pi = ms.argmax(2)
        ti = mt.argmax(2)
        ok = (use.sum(2) > 0) & valid_cell
        self.top1_ok += int(((pi == ti) & ok).sum().detach().cpu())
        self.top1_n += int(ok.sum().detach().cpu())
        chosen = torch.gather(target, 2, pi[:, :, None, :]).squeeze(2)
        best = mt.max(2).values
        if torch.any(ok):
            cv = chosen[ok]
            bv = best[ok]
            self.selected_n += int(cv.numel())
            self.selected_beneficial += int((cv >= float(self.beneficial_threshold)).sum().detach().cpu())
            self.selected_harmful += int((cv < 0.0).sum().detach().cpu())
            self.selected_strong_harm += int((cv <= -float(self.harmful_threshold)).sum().detach().cpu())
            self.oracle_beneficial += int((bv >= float(self.beneficial_threshold)).sum().detach().cpu())
            self.selected_utility_sum += float(cv.sum().detach().cpu())
            self.regret_sum += float((bv - cv).sum().detach().cpu())

        hi = mt.argmax(2)
        lo = target.masked_fill(~use, 1e6).argmin(2)
        sp = torch.gather(score, 2, hi[:, :, None, :]).squeeze(2)
        sn = torch.gather(score, 2, lo[:, :, None, :]).squeeze(2)
        tp = torch.gather(target, 2, hi[:, :, None, :]).squeeze(2)
        tn = torch.gather(target, 2, lo[:, :, None, :]).squeeze(2)
        enough = (use.sum(2) >= 2) & ((tp - tn) >= 0.02) & valid_cell
        self.pair_ok += int(((sp > sn) & enough).sum().detach().cpu())
        self.pair_n += int(enough.sum().detach().cpu())

        if torch.any(use):
            s = score[use].detach().cpu().numpy()
            y = target[use].detach().cpu().numpy()
            take = min(len(s), int(self.sample_cap_per_update))
            if take:
                rng = np.random.default_rng(self.seed + self.updates * 997)
                ids = rng.choice(len(s), size=take, replace=False) if len(s) > take else np.arange(len(s))
                self.score_samples.append(s[ids].astype(np.float32, copy=False))
                self.target_samples.append(y[ids].astype(np.float32, copy=False))
        self.updates += 1

    def summary(self):
        out = {
            'cell_count': int(self.selected_n),
            'pairwise_rank_count': int(self.pair_n),
            'pairwise_rank_accuracy': float(self.pair_ok / max(self.pair_n, 1)),
            'top1_agreement': float(self.top1_ok / max(self.top1_n, 1)),
            'selected_beneficial_rate': float(self.selected_beneficial / max(self.selected_n, 1)),
            'selected_harmful_rate': float(self.selected_harmful / max(self.selected_n, 1)),
            'selected_strong_harm_rate': float(self.selected_strong_harm / max(self.selected_n, 1)),
            'oracle_any_beneficial_rate': float(self.oracle_beneficial / max(self.selected_n, 1)),
            'beneficial_capture_ratio': float(self.selected_beneficial / max(self.oracle_beneficial, 1)),
            'mean_selected_relative_utility': float(self.selected_utility_sum / max(self.selected_n, 1)),
            'mean_selection_regret': float(self.regret_sum / max(self.selected_n, 1)),
        }
        if self.score_samples:
            s = np.concatenate(self.score_samples)
            y = np.concatenate(self.target_samples)
            out['beneficial_auroc'] = binary_auroc(
                s, y >= float(self.beneficial_threshold), max_points=300000, seed=self.seed + 13
            )
            out['sampled_candidate_cells'] = int(len(s))
        else:
            out['beneficial_auroc'] = None
            out['sampled_candidate_cells'] = 0
        return out


def safe_selection_score(mu, sigma, beneficial_logit, beta=0.5, benefit_logit_weight=0.25):
    """Deployment-safe support score used by v0.9A.1.

    Relative utility handles ordering, the beneficial logit supplies an absolute
    sign cue, and sigma penalizes uncertain support.  All three terms are produced
    from deployment-safe inputs.
    """
    return mu + float(benefit_logit_weight) * beneficial_logit - float(beta) * sigma


def _listwise_best_loss(score, target, use, beneficial_threshold, temperature=0.2):
    """Directly train the best beneficial candidate to occupy rank 1.

    Cells with no beneficial candidate are excluded; those cells are handled by
    the absolute sign/barrier objective and later by the no-support policy.
    """
    # [B,N,K,P] -> [B,N,P,K]
    logits = score.permute(0, 1, 3, 2)
    tgt = target.permute(0, 1, 3, 2)
    use_k = use.permute(0, 1, 3, 2)
    masked_tgt = tgt.masked_fill(~use_k, -1e6)
    best_val, best_idx = masked_tgt.max(-1)
    eligible = (use_k.sum(-1) >= 2) & (best_val >= float(beneficial_threshold))
    if not torch.any(eligible):
        z = score.sum() * 0.0
        return z, torch.tensor(0.0, device=score.device), 0
    masked_logits = logits.masked_fill(~use_k, -1e4) / float(temperature)
    loss = F.cross_entropy(masked_logits[eligible], best_idx[eligible])
    pred = masked_logits.argmax(-1)
    acc = (pred[eligible] == best_idx[eligible]).float().mean()
    return loss, acc, int(eligible.sum().detach().cpu())


def _focal_sign_loss(logit, target, use, beneficial_threshold, harmful_threshold, gamma=1.5):
    clear = use & _clear_mask(target, beneficial_threshold, harmful_threshold)
    if not torch.any(clear):
        return logit.sum() * 0.0
    label = (target >= float(beneficial_threshold)).float()
    bce = F.binary_cross_entropy_with_logits(logit[clear], label[clear], reduction='none')
    prob = torch.sigmoid(logit[clear])
    pt = torch.where(label[clear] > 0.5, prob, 1.0 - prob)
    return (((1.0 - pt).clamp_min(1e-6) ** float(gamma)) * bce).mean()


def _absolute_score_barrier(score, target, use, beneficial_threshold, harmful_threshold,
                            positive_margin=0.08, harmful_margin=-0.08):
    pos = use & (target >= float(beneficial_threshold))
    neg = use & (target <= -float(harmful_threshold))
    terms = []
    if torch.any(pos):
        terms.append(torch.relu(float(positive_margin) - score[pos]).mean())
    if torch.any(neg):
        terms.append(torch.relu(score[neg] - float(harmful_margin)).mean())
    if not terms:
        return score.sum() * 0.0
    return torch.stack(terms).mean()


def predictive_utility_safe_loss(mu, sigma, beneficial_logit, target, candidate_mask, valid_cell,
                                 coarse_score, cfg):
    """v0.9A.1 objective focused on safe top selection rather than pairwise rank alone."""
    c = cfg['predictive_utility_safe_selection']['training']
    use = candidate_mask[:, :, :, None] & valid_cell[:, :, None, :]
    if not torch.any(use):
        z = mu.sum() * 0.0
        return z, {
            'utility': z, 'rank': z, 'benefit': z, 'hard': z, 'listwise': z, 'barrier': z,
            'rank_acc': 0.0, 'hard_acc': 0.0, 'top1_acc': 0.0,
            'rank_n': 0, 'hard_n': 0, 'top1_n': 0,
        }

    huber = F.smooth_l1_loss(mu, target, beta=float(c['huber_beta']), reduction='none')
    reg_raw = huber / sigma.clamp_min(1e-4) + torch.log(sigma.clamp_min(1e-4))
    utility = reg_raw[use].mean()

    select_score = safe_selection_score(
        mu, sigma, beneficial_logit,
        beta=float(c['train_risk_beta']),
        benefit_logit_weight=float(c['train_benefit_logit_weight']),
    )
    rank, rank_acc, rank_n = _pairwise_best_worst(
        select_score, target, use, float(c['rank_margin']), float(c['rank_min_gap'])
    )
    benefit = _focal_sign_loss(
        beneficial_logit, target, use,
        float(c['beneficial_threshold']), float(c['harmful_threshold']),
        gamma=float(c['focal_gamma']),
    )
    hard, hard_acc, hard_n = _hard_harmful_rank(
        select_score, target, use, coarse_score,
        float(c['beneficial_threshold']), float(c['harmful_threshold']),
        float(c['hard_margin']), float(c['hard_similarity_weight']), float(c['hard_utility_weight'])
    )
    listwise, top1_acc, top1_n = _listwise_best_loss(
        select_score, target, use, float(c['beneficial_threshold']), float(c['listwise_temperature'])
    )
    barrier = _absolute_score_barrier(
        select_score, target, use,
        float(c['beneficial_threshold']), float(c['harmful_threshold']),
        float(c['barrier_positive_margin']), float(c['barrier_harmful_margin']),
    )

    loss = (
        float(c['utility_weight']) * utility
        + float(c['rank_weight']) * rank
        + float(c['classification_weight']) * benefit
        + float(c['hard_harmful_weight']) * hard
        + float(c['listwise_weight']) * listwise
        + float(c['barrier_weight']) * barrier
    )
    return loss, {
        'utility': utility, 'rank': rank, 'benefit': benefit, 'hard': hard,
        'listwise': listwise, 'barrier': barrier,
        'rank_acc': float(rank_acc.detach().cpu()),
        'hard_acc': float(hard_acc.detach().cpu()),
        'top1_acc': float(top1_acc.detach().cpu()),
        'rank_n': rank_n, 'hard_n': hard_n, 'top1_n': top1_n,
    }


@dataclass
class SafePolicyAccumulator:
    """Metrics for a policy that may abstain and retain the protected backbone."""
    beneficial_threshold: float
    harmful_threshold: float
    threshold: float
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

    def update(self, score, target, candidate_mask, valid_cell):
        use = candidate_mask[:, :, :, None] & valid_cell[:, :, None, :]
        ms = score.masked_fill(~use, -1e6)
        mt = target.masked_fill(~use, -1e6)
        pi = ms.argmax(2)
        top_score = ms.max(2).values
        chosen = torch.gather(target, 2, pi[:, :, None, :]).squeeze(2)
        best = mt.max(2).values
        ok = (use.sum(2) > 0) & valid_cell
        accept = ok & (top_score >= float(self.threshold))
        if not torch.any(ok):
            return
        cv = chosen[ok]
        bv = best[ok]
        av = accept[ok]
        policy = torch.where(av, cv, torch.zeros_like(cv))
        oracle_safe = torch.clamp(bv, min=0.0)
        self.total_n += int(cv.numel())
        self.accepted_n += int(av.sum().detach().cpu())
        if torch.any(av):
            cav = cv[av]
            self.accepted_beneficial += int((cav >= float(self.beneficial_threshold)).sum().detach().cpu())
            self.accepted_harmful += int((cav < 0.0).sum().detach().cpu())
            self.accepted_strong_harm += int((cav <= -float(self.harmful_threshold)).sum().detach().cpu())
            self.accepted_utility_sum += float(cav.sum().detach().cpu())
        self.oracle_beneficial += int((bv >= float(self.beneficial_threshold)).sum().detach().cpu())
        self.policy_utility_sum += float(policy.sum().detach().cpu())
        self.safe_oracle_utility_sum += float(oracle_safe.sum().detach().cpu())
        self.safe_regret_sum += float((oracle_safe - policy).sum().detach().cpu())

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
            'threshold': float(self.threshold),
        }


def safe_policy_metrics_from_records(top_score, chosen_utility, best_utility, threshold,
                                     beneficial_threshold, harmful_threshold):
    s = np.asarray(top_score, np.float64)
    u = np.asarray(chosen_utility, np.float64)
    b = np.asarray(best_utility, np.float64)
    accept = s >= float(threshold)
    n = len(s)
    na = int(accept.sum())
    au = u[accept]
    policy = np.where(accept, u, 0.0)
    oracle_safe = np.maximum(b, 0.0)
    oracle_b = b >= float(beneficial_threshold)
    return {
        'cell_count': int(n),
        'coverage': float(na / max(n, 1)),
        'accepted_count': na,
        'accepted_beneficial_rate': float(np.mean(au >= float(beneficial_threshold))) if na else 0.0,
        'accepted_harmful_rate': float(np.mean(au < 0.0)) if na else 0.0,
        'accepted_strong_harm_rate': float(np.mean(au <= -float(harmful_threshold))) if na else 0.0,
        'beneficial_capture_ratio': float(np.sum(accept & (u >= float(beneficial_threshold))) / max(int(oracle_b.sum()), 1)),
        'mean_policy_relative_utility': float(np.mean(policy)) if n else 0.0,
        'mean_accepted_relative_utility': float(np.mean(au)) if na else 0.0,
        'mean_safe_oracle_relative_utility': float(np.mean(oracle_safe)) if n else 0.0,
        'mean_safe_policy_regret': float(np.mean(oracle_safe - policy)) if n else 0.0,
        'threshold': float(threshold),
    }


def calibrate_abstain_threshold(top_score, chosen_utility, best_utility, cfg):
    """Choose a Train-dev-only no-support threshold with an explicit coverage floor."""
    p = cfg['predictive_utility_safe_selection']['policy']
    t = cfg['predictive_utility_safe_selection']['training']
    scores = np.asarray(top_score, np.float64)
    if scores.size == 0:
        raise RuntimeError('No Train-dev cells available for abstention calibration')
    best = None
    rows = []
    for coverage in [float(x) for x in p['coverage_grid']]:
        if coverage >= 0.999999:
            threshold = float(np.min(scores) - 1e-6)
        else:
            threshold = float(np.quantile(scores, 1.0 - coverage))
        m = safe_policy_metrics_from_records(
            scores, chosen_utility, best_utility, threshold,
            float(t['beneficial_threshold']), float(t['harmful_threshold']),
        )
        harmful_exposure = m['coverage'] * m['accepted_harmful_rate']
        strong_exposure = m['coverage'] * m['accepted_strong_harm_rate']
        objective = (
            m['mean_policy_relative_utility']
            + float(p['capture_reward']) * m['beneficial_capture_ratio']
            - float(p['harmful_exposure_penalty']) * harmful_exposure
            - float(p['strong_harm_exposure_penalty']) * strong_exposure
        )
        feasible = m['coverage'] + 1e-9 >= float(p['min_coverage'])
        row = dict(m)
        row['requested_coverage'] = coverage
        row['objective'] = float(objective)
        row['feasible'] = bool(feasible)
        rows.append(row)
        item = (1 if feasible else 0, float(objective), -m['accepted_strong_harm_rate'],
                -m['accepted_harmful_rate'], m['coverage'])
        if best is None or item > best[0]:
            best = (item, row)
    return best[1], rows
