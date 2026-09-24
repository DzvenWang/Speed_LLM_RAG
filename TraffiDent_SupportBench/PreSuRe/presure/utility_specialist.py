from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import torch

from .features import NODE_SENSE_DIM
from .need_gated import RankAccumulator, gather_support_context, sense_features, utility_targets, quantile_threshold
from .probes import binary_auroc


def true_q90_mask(base_raw, truth_raw, q90_thresholds_raw):
    """Return privileged Train/diagnostic q90 hard-error mask [B,P,N].

    q90_thresholds_raw is horizon-specific and must be estimated from Train only.
    The returned mask is supervision/evaluation only and is never a deployable input.
    """
    b = np.asarray(base_raw, np.float32)
    y = np.asarray(truth_raw, np.float32)
    q = np.asarray(q90_thresholds_raw, np.float32).reshape(1, -1, 1)
    if b.shape != y.shape or b.ndim != 3 or b.shape[1] != q.shape[1]:
        raise ValueError(f'incompatible base/truth/q90 shapes: {b.shape} {y.shape} {q.shape}')
    return np.abs(y - b) >= q


def utility_specialist_batch(
    model,
    node_q,
    node_mem,
    residual_mem,
    base_raw,
    truth_raw,
    valid_raw,
    src_ids,
    top4_ids,
    top4_conf,
    top4_mask,
    need_prob,
    mean,
    std,
    history,
    horizon,
    cfg,
    device,
    condition_on_need=True,
):
    """Build one Utility-Specialist batch.

    Current-query future is used only for the Train utility target.  The model input
    contains no current future.  `need_prob` is the deployable v0.7.2 q90 probability.
    """
    qctx_np = np.asarray(node_q[src_ids], np.float32)
    gids = np.asarray(top4_ids, np.int64)
    cctx_np, rr_raw_np = gather_support_context(node_mem, residual_mem, gids)

    qctx = torch.as_tensor(np.array(qctx_np, copy=True), device=device, dtype=torch.float32)
    cctx = torch.as_tensor(np.array(cctx_np, copy=True), device=device, dtype=torch.float32)
    rr_raw = torch.as_tensor(np.array(rr_raw_np, copy=True), device=device, dtype=torch.float32)
    rr = rr_raw / float(std)
    qs = torch.as_tensor(np.array(sense_features(qctx_np), copy=True), device=device, dtype=torch.float32)
    ss = torch.as_tensor(np.array(sense_features(cctx_np), copy=True), device=device, dtype=torch.float32)
    conf = torch.as_tensor(np.array(top4_conf, copy=True), device=device, dtype=torch.float32)
    cm = torch.as_tensor(np.array(top4_mask, copy=True), device=device, dtype=torch.bool)
    b = torch.as_tensor(np.array(base_raw[src_ids], copy=True), device=device, dtype=torch.float32)
    y = torch.as_tensor(np.array(truth_raw[src_ids], copy=True), device=device, dtype=torch.float32)
    qbn = (b - float(mean)) / float(std)
    sbase = cctx[..., 4 * int(history) : 4 * int(history) + int(horizon)]
    true_need = (y - b) / float(std)
    np_need = np.asarray(need_prob, np.float32)
    need = torch.as_tensor(np.array(np_need, copy=True), device=device, dtype=torch.float32)
    if not condition_on_need:
        need = torch.zeros_like(need)
    pred = model(qctx, cctx, qs, ss, rr, qbn, sbase, conf, need)
    target = utility_targets(true_need, rr, cm, cfg)
    valid = torch.as_tensor(np.array(valid_raw[src_ids], copy=True), device=device, dtype=torch.bool)
    return pred, target, cm, valid, rr_raw


def specialist_loss(pred, target, candidate_mask, valid, true_hard, need_prob, cfg, mode):
    """Pure support-discrimination objective for v0.7.4.

    No utility regression, support-mass or forecast loss is used.  We optimize only:
      1) clear beneficial-vs-harmful candidate classification; and
      2) pairwise best-vs-worst ranking.

    mode='hard' uses only Train true-q90 cells.
    mode='weighted' uses all Train cells but weights them by deployable Need-q90.
    """
    c = cfg['need_conditioned_utility_specialist_probe']['training']
    vm = valid.bool().permute(0, 2, 1)  # [B,N,P]
    hard = true_hard.bool().permute(0, 2, 1)
    npb = need_prob.float().permute(0, 2, 1).clamp(0.0, 1.0)
    if mode == 'hard':
        cell = vm & hard
        cell_w = torch.ones_like(npb)
    elif mode == 'weighted':
        cell = vm
        cell_w = 1.0 + float(c['need_weight_lambda']) * npb
    else:
        raise ValueError(mode)

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
        cw = cell_w[:, :, None, :].expand_as(pred)
        cls = (raw[clear] * cw[clear]).sum() / cw[clear].sum().clamp_min(1.0)
    else:
        cls = pred.sum() * 0.0

    masked_t = target.masked_fill(~use, -1e6)
    hi = masked_t.argmax(2)
    lo = target.masked_fill(~use, 1e6).argmin(2)
    sp = torch.gather(pred, 2, hi[:, :, None, :]).squeeze(2)
    sn = torch.gather(pred, 2, lo[:, :, None, :]).squeeze(2)
    tp = torch.gather(target, 2, hi[:, :, None, :]).squeeze(2)
    tn = torch.gather(target, 2, lo[:, :, None, :]).squeeze(2)
    enough = (use.sum(2) >= 2) & ((tp - tn) >= float(c['rank_min_gap'])) & cell
    if torch.any(enough):
        raw_rank = torch.relu(float(c['rank_margin']) - sp + sn)
        rw = cell_w
        rank = (raw_rank[enough] * rw[enough]).sum() / rw[enough].sum().clamp_min(1.0)
        racc = (sp[enough] > sn[enough]).float().mean()
        rank_n = int(enough.sum().detach().cpu())
    else:
        rank = pred.sum() * 0.0
        racc = torch.tensor(0.0, device=pred.device)
        rank_n = 0

    loss = float(c['rank_weight']) * rank + float(c['classification_weight']) * cls
    return loss, rank, cls, racc, rank_n


def probability_score(logit, temperature):
    return torch.sigmoid(logit / float(temperature))


def two_expert_score(generic_score, hard_score, need_prob, need_threshold, hard_mix_weight, temperature):
    """Deployment-safe General/Hard mixture.

    Low-Need cells use Generic Utility-A. High-Need cells interpolate toward the
    hard-q90 specialist. Scores are converted to beneficial probabilities first so
    model-scale differences do not contaminate the mixture.
    """
    g = probability_score(generic_score, temperature)
    h = probability_score(hard_score, temperature)
    p = need_prob.float().permute(0, 2, 1)[:, :, None, :]
    route = (p >= float(need_threshold)).float()
    w = route * float(hard_mix_weight)
    return (1.0 - w) * g + w * h


@dataclass
class CandidateMetricAccumulator:
    beneficial_threshold: float
    sample_cap_per_update: int = 4096
    seed: int = 10
    rank: RankAccumulator = field(default_factory=RankAccumulator)
    score_samples: list = field(default_factory=list)
    target_samples: list = field(default_factory=list)
    updates: int = 0

    def update(self, score, target, candidate_mask, valid_cell):
        self.rank.update(score, target, candidate_mask, valid_cell, self.beneficial_threshold)
        use = candidate_mask[:, :, :, None] & valid_cell[:, :, None, :]
        if torch.any(use):
            s = score[use].detach().cpu().numpy()
            y = target[use].detach().cpu().numpy()
            take = min(len(s), int(self.sample_cap_per_update))
            if take > 0:
                rng = np.random.default_rng(self.seed + self.updates * 997)
                ids = rng.choice(len(s), size=take, replace=False) if len(s) > take else np.arange(len(s))
                self.score_samples.append(s[ids].astype(np.float32, copy=False))
                self.target_samples.append(y[ids].astype(np.float32, copy=False))
        self.updates += 1

    def summary(self):
        out = self.rank.summary()
        if self.score_samples:
            s = np.concatenate(self.score_samples)
            y = np.concatenate(self.target_samples)
            out['beneficial_auroc'] = binary_auroc(s, y > float(self.beneficial_threshold), max_points=300000, seed=self.seed + 17)
            out['sampled_candidate_cells'] = int(len(s))
        else:
            out['beneficial_auroc'] = None
            out['sampled_candidate_cells'] = 0
        return out


def predicted_need_threshold(need_prob, valid_mask, keep_fraction):
    return quantile_threshold(need_prob, valid_mask, keep_fraction)
