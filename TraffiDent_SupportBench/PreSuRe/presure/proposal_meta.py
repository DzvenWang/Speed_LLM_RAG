from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn


class SupportProposalMetaRanker(nn.Module):
    """Score support proposals produced by frozen utility experts.

    The input is entirely deployment safe: predicted Need, expert scores/confidence,
    M3 confidence, historical support residual summaries, expert agreement, and
    horizon identity.  The current-query future is used only to construct Train
    utility targets outside this module.
    """
    def __init__(self, input_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, int(hidden)), nn.GELU(), nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(hidden)), nn.GELU(), nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _top2(x: torch.Tensor, mask: torch.Tensor):
    """Return top1/top2/margin/index along candidate dimension L.

    x: [B,N,L,P], mask: [B,N,L]
    """
    m = mask[:, :, :, None].expand_as(x)
    vals, idx = torch.sort(x.masked_fill(~m, -1e6), dim=2, descending=True)
    top1 = vals[:, :, 0, :]
    top2 = vals[:, :, 1, :] if vals.shape[2] > 1 else torch.zeros_like(top1)
    return top1, top2, top1 - top2, idx[:, :, 0, :]


def build_support_proposals(
    expert_scores,
    expert_order,
    target,
    rr_raw,
    m3_conf,
    candidate_mask,
    need_prob,
    temperature,
    residual_scale,
    query_context=None,
    support_context=None,
):
    """Merge frozen experts' Top1 supports into <=E unique proposals per cell.

    query_context [B,N,F] and support_context [B,N,L,F] are optional deployment-safe
    node representations. When supplied, query/support/difference context is appended.

    Returns a dict with:
      features      [B,N,P,S,F]
      target        [B,N,P,S] privileged Train/eval utility only
      mask          [B,N,P,S]
      candidate_idx [B,N,P,S] index into the original M3 Top-K list
      agreement     [B,N,P,S]
      unique_count  [B,N,P]

    Duplicate proposals are represented once (first expert in expert_order wins the
    slot) while the feature vector records every expert that proposed the support.
    """
    order = list(expert_order)
    E = len(order)
    if E < 2:
        raise ValueError('proposal meta-ranking needs >=2 experts')
    B, N, L, P = target.shape
    temp = float(temperature)
    scale = max(float(residual_scale), 1e-6)

    probs = {k: torch.sigmoid(expert_scores[k] / temp) for k in order}
    top1_prob, margin, top_idx = [], [], []
    for k in order:
        a, _, m, ix = _top2(probs[k], candidate_mask)
        top1_prob.append(a); margin.append(m); top_idx.append(ix)
    top1_prob = torch.stack(top1_prob, dim=-1)  # [B,N,P,E]
    margin = torch.stack(margin, dim=-1)
    top_idx = torch.stack(top_idx, dim=-1)      # [B,N,P,E]

    # M3 shared summaries.
    cm = candidate_mask.bool()
    cmf = cm.float()
    cnt = cmf.sum(2).clamp_min(1.0)
    m3 = m3_conf.float()
    m3s, _ = torch.sort(m3.masked_fill(~cm, -1e6), dim=2, descending=True)
    m3_1 = m3s[:, :, 0].clamp_min(0.0)
    m3_2 = m3s[:, :, 1].clamp_min(0.0) if L > 1 else torch.zeros_like(m3_1)
    m3_mean = (m3.masked_fill(~cm, 0.0).sum(2) / cnt).clamp_min(0.0)
    need = need_prob.float().permute(0, 2, 1)  # [B,N,P]
    hz = torch.linspace(0.0, 1.0, P, device=target.device).view(1, 1, P).expand(B, N, P)
    available = cm.any(2)[:, :, None].expand(B, N, P)

    feat_slots, tar_slots, mask_slots, idx_slots, agree_slots = [], [], [], [], []
    all_prop_eq = []
    for s in range(E):
        ix = top_idx[..., s]  # [B,N,P]
        eq = top_idx == ix[..., None]  # [B,N,P,E]
        all_prop_eq.append(eq)
        unique = available.clone()
        if s > 0:
            duplicate = (top_idx[..., :s] == ix[..., None]).any(-1)
            unique &= ~duplicate

        # Gather target and residual trajectory of this proposal.
        g = ix[:, :, None, :]
        ut = torch.gather(target, 2, g).squeeze(2)  # [B,N,P]
        # rr_raw is [B,N,L,Ptrajectory]. Gather candidate -> [B,N,Pcell,Ptraj].
        rr5 = rr_raw[:, :, :, None, :].expand(-1, -1, -1, P, -1)
        rr = torch.gather(
            rr5,
            2,
            ix[:, :, None, :, None].expand(-1, -1, 1, -1, P),
        ).squeeze(2)
        # The gather above treats the proposal-cell horizon as an extra axis; after
        # broadcasting the stored trajectory is indexed as [B,N,Pcell,Ptraj].
        local_rr = torch.diagonal(rr, dim1=2, dim2=3)  # [B,N,P]
        rrn = rr / scale
        local_rrn = local_rr / scale
        rr_mean = rrn.mean(-1)
        rr_std = rrn.std(-1, unbiased=False)
        rr_maxabs = rrn.abs().amax(-1)

        # Probability assigned to this candidate by every expert.
        cand_prob = []
        for k in order:
            cand_prob.append(torch.gather(probs[k], 2, g).squeeze(2))
        cand_prob = torch.stack(cand_prob, dim=-1)  # [B,N,P,E]

        # M3 confidence for proposed candidate is horizon invariant.
        # ix can differ by horizon, so gather with a P-expanded confidence tensor.
        m3e = m3[:, :, :, None].expand(-1, -1, -1, P)
        m3_cand = torch.gather(m3e, 2, g).squeeze(2)
        agreement = eq.float().sum(-1)

        # Unique count is filled after all slots are formed; use a placeholder now.
        shared_top = torch.cat([top1_prob, margin], dim=-1)  # 2E
        pieces = [
            cand_prob,                      # E
            eq.float(),                     # E proposer indicators
            shared_top,                     # 2E expert top1 probs + margins
            need[..., None],                # 1
            m3_cand[..., None],             # 1
            m3_1[:, :, None].expand(-1,-1,P)[...,None],
            m3_2[:, :, None].expand(-1,-1,P)[...,None],
            (m3_1-m3_2)[:, :, None].expand(-1,-1,P)[...,None],
            m3_mean[:, :, None].expand(-1,-1,P)[...,None],
            (agreement / float(E))[...,None],
            local_rrn[...,None], local_rrn.abs()[...,None],
            rr_mean[...,None], rr_std[...,None], rr_maxabs[...,None],
            hz[...,None],
        ]
        if query_context is not None or support_context is not None:
            if query_context is None or support_context is None:
                raise ValueError('query_context and support_context must be supplied together')
            qx = query_context.float()[:, :, None, :].expand(-1, -1, P, -1)
            sx5 = support_context.float()[:, :, :, None, :].expand(-1, -1, -1, P, -1)
            sx = torch.gather(
                sx5, 2, ix[:, :, None, :, None].expand(-1, -1, 1, -1, sx5.shape[-1])
            ).squeeze(2)
            pieces.extend([qx, sx, torch.abs(qx - sx)])
        f = torch.cat(pieces, dim=-1)
        feat_slots.append(f)
        tar_slots.append(ut)
        mask_slots.append(unique)
        idx_slots.append(ix)
        agree_slots.append(agreement)

    features = torch.stack(feat_slots, dim=3)  # [B,N,P,S,F0]
    targets = torch.stack(tar_slots, dim=3)
    pmask = torch.stack(mask_slots, dim=3)
    pidx = torch.stack(idx_slots, dim=3)
    agreement = torch.stack(agree_slots, dim=3)
    unique_count = pmask.sum(3).float()
    # Append unique-proposal-count, a cell-level disagreement indicator.
    uniq = (unique_count / float(E))[..., None, None].expand(-1,-1,-1,E,1)
    features = torch.cat([features, uniq], dim=-1)
    return {
        'features': features,
        'target': targets,
        'mask': pmask,
        'candidate_idx': pidx,
        'agreement': agreement,
        'unique_count': unique_count,
    }


def meta_ranker_loss(score, target, proposal_mask, valid, need_prob, cfg):
    """Regret-aware proposal objective; ambiguous cells are never discarded."""
    c = cfg['proposal_meta_ranking_probe']['training']
    vm = valid.bool().permute(0, 2, 1)  # [B,N,P]
    use = proposal_mask.bool() & vm[..., None]
    if not torch.any(use):
        z = score.sum() * 0.0
        return z, {'rank': 0.0, 'utility': 0.0, 'benefit': 0.0, 'regret': 0.0, 'rank_acc': 0.0, 'cells': 0}

    need = need_prob.float().permute(0, 2, 1).clamp(0.0, 1.0)
    cell_w = 1.0 + float(c['need_weight_lambda']) * need
    prop_w = cell_w[..., None].expand_as(score)

    # Utility regression on every valid proposal.
    reg_raw = torch.nn.functional.smooth_l1_loss(score, target, reduction='none', beta=float(c['huber_beta']))
    reg = (reg_raw[use] * prop_w[use]).sum() / prop_w[use].sum().clamp_min(1.0)

    # Clear beneficial/harmful classification via the same scalar score.
    clear = use & ((target >= float(c['clear_beneficial_threshold'])) | (target <= -float(c['clear_harm_threshold'])))
    if torch.any(clear):
        label = (target >= float(c['clear_beneficial_threshold'])).float()
        braw = torch.nn.functional.binary_cross_entropy_with_logits(
            score / float(c['classification_temperature']), label, reduction='none'
        )
        bce = (braw[clear] * prop_w[clear]).sum() / prop_w[clear].sum().clamp_min(1.0)
    else:
        bce = score.sum() * 0.0

    # Best-vs-worst proposal ranking, only when a real target gap exists.
    mt_hi = target.masked_fill(~use, -1e6)
    mt_lo = target.masked_fill(~use, 1e6)
    hi = mt_hi.argmax(-1); lo = mt_lo.argmin(-1)
    sp = torch.gather(score, -1, hi[..., None]).squeeze(-1)
    sn = torch.gather(score, -1, lo[..., None]).squeeze(-1)
    tp = torch.gather(target, -1, hi[..., None]).squeeze(-1)
    tn = torch.gather(target, -1, lo[..., None]).squeeze(-1)
    enough = vm & (use.sum(-1) >= 2) & ((tp - tn) >= float(c['rank_min_gap']))
    if torch.any(enough):
        rraw = torch.relu(float(c['rank_margin']) - sp + sn)
        rank = (rraw[enough] * cell_w[enough]).sum() / cell_w[enough].sum().clamp_min(1.0)
        rank_acc = (sp[enough] > sn[enough]).float().mean()
        cells = int(enough.sum().detach().cpu())
    else:
        rank = score.sum() * 0.0
        rank_acc = torch.tensor(0.0, device=score.device)
        cells = 0

    # Expected regret. No sample is discarded merely because experts are close.
    masked_score = score.masked_fill(~use, -1e6)
    p = torch.softmax(masked_score / float(c['regret_temperature']), dim=-1)
    best = mt_hi.max(-1).values
    regret_each = (best[..., None] - target).clamp_min(0.0)
    exp_regret = (p * regret_each.masked_fill(~use, 0.0)).sum(-1)
    has = vm & (use.sum(-1) > 0)
    if torch.any(has):
        regret = (exp_regret[has] * cell_w[has]).sum() / cell_w[has].sum().clamp_min(1.0)
    else:
        regret = score.sum() * 0.0

    loss = (
        float(c['rank_weight']) * rank
        + float(c['utility_weight']) * reg
        + float(c['classification_weight']) * bce
        + float(c['regret_weight']) * regret
    )
    return loss, {
        'rank': float(rank.detach()), 'utility': float(reg.detach()),
        'benefit': float(bce.detach()), 'regret': float(regret.detach()),
        'rank_acc': float(rank_acc.detach()), 'cells': cells,
    }


@dataclass
class ProposalMetricAccumulator:
    beneficial_threshold: float
    near_oracle_tolerance: float = 0.02
    regret_sample_cap_per_update: int = 4096
    seed: int = 10
    selected_n: int = 0
    selected_beneficial: int = 0
    selected_harmful: int = 0
    selected_strong_harm: int = 0
    oracle_available: int = 0
    top1_ok: int = 0
    rank_ok: int = 0
    rank_n: int = 0
    near_oracle: int = 0
    harmful_regret: int = 0
    chosen_utility_sum: float = 0.0
    oracle_utility_sum: float = 0.0
    regret_sum: float = 0.0
    regret_samples: list = field(default_factory=list)
    updates: int = 0

    def update(self, score, target, proposal_mask, valid_cell):
        use = proposal_mask.bool() & valid_cell[..., None]
        ms = score.masked_fill(~use, -1e6)
        mt = target.masked_fill(~use, -1e6)
        pi = ms.argmax(-1); ti = mt.argmax(-1)
        ok = valid_cell & (use.sum(-1) > 0)
        if not torch.any(ok):
            return
        chosen = torch.gather(target, -1, pi[..., None]).squeeze(-1)
        best = mt.max(-1).values
        regret = (best - chosen).clamp_min(0.0)
        cv = chosen[ok]; bv = best[ok]; rv = regret[ok]
        self.selected_n += int(cv.numel())
        self.selected_beneficial += int((cv > float(self.beneficial_threshold)).sum().cpu())
        self.selected_harmful += int((cv < 0.0).sum().cpu())
        self.selected_strong_harm += int((cv < -float(self.beneficial_threshold)).sum().cpu())
        self.oracle_available += int((bv > float(self.beneficial_threshold)).sum().cpu())
        self.top1_ok += int(((pi == ti) & ok).sum().cpu())
        self.near_oracle += int(((regret <= float(self.near_oracle_tolerance)) & ok).sum().cpu())
        self.harmful_regret += int(((chosen < 0.0) & (best > float(self.beneficial_threshold)) & ok).sum().cpu())
        self.chosen_utility_sum += float(cv.sum().cpu())
        self.oracle_utility_sum += float(bv.sum().cpu())
        self.regret_sum += float(rv.sum().cpu())

        lo = target.masked_fill(~use, 1e6).argmin(-1)
        sp = torch.gather(score, -1, ti[..., None]).squeeze(-1)
        sn = torch.gather(score, -1, lo[..., None]).squeeze(-1)
        tp = torch.gather(target, -1, ti[..., None]).squeeze(-1)
        tn = torch.gather(target, -1, lo[..., None]).squeeze(-1)
        enough = valid_cell & (use.sum(-1) >= 2) & ((tp - tn) > 0.02)
        self.rank_ok += int(((sp > sn) & enough).sum().cpu())
        self.rank_n += int(enough.sum().cpu())

        arr = rv.detach().cpu().numpy()
        take = min(len(arr), int(self.regret_sample_cap_per_update))
        if take:
            rng = np.random.default_rng(self.seed + self.updates * 1009)
            ids = rng.choice(len(arr), take, replace=False) if len(arr) > take else np.arange(len(arr))
            self.regret_samples.append(arr[ids].astype(np.float32, copy=False))
        self.updates += 1

    def summary(self):
        n = max(self.selected_n, 1)
        r = np.concatenate(self.regret_samples) if self.regret_samples else np.empty((0,), np.float32)
        return {
            'cell_count': int(self.selected_n),
            'pairwise_rank_count': int(self.rank_n),
            'pairwise_rank_accuracy': float(self.rank_ok / max(self.rank_n, 1)),
            'top1_agreement': float(self.top1_ok / n),
            'selected_beneficial_rate': float(self.selected_beneficial / n),
            'selected_harmful_rate': float(self.selected_harmful / n),
            'selected_strong_harm_rate': float(self.selected_strong_harm / n),
            'oracle_any_beneficial_rate': float(self.oracle_available / n),
            'beneficial_capture_ratio': float(self.selected_beneficial / max(self.oracle_available, 1)),
            'mean_selected_relative_utility': float(self.chosen_utility_sum / n),
            'mean_oracle_proposal_utility': float(self.oracle_utility_sum / n),
            'mean_selection_regret': float(self.regret_sum / n),
            'median_selection_regret_sampled': None if len(r) == 0 else float(np.median(r)),
            'p90_selection_regret_sampled': None if len(r) == 0 else float(np.quantile(r, 0.90)),
            'near_oracle_rate': float(self.near_oracle / n),
            'harmful_regret_rate': float(self.harmful_regret / n),
        }
