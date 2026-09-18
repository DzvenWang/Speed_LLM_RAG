from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import torch

from .features import NODE_SENSE_DIM
from .probes import SupportUtilityProbeNet


def sense_features(x):
    return np.asarray(x[..., -NODE_SENSE_DIM:], np.float32)


def gather_support_context(node_mem, residual_mem, gids):
    """Gather same-node historical support context/residuals.

    node_mem: [T,N,F]
    residual_mem: [T,P,N] in raw speed units
    gids: [B,N,L] historical query ids
    returns context [B,N,L,F], residual [B,N,L,P]
    """
    gids = np.asarray(gids, np.int64)
    N = gids.shape[1]
    ni = np.arange(N, dtype=np.int64)[None, :, None]
    ctx = np.asarray(node_mem)[gids, ni, :].astype(np.float32, copy=False)
    rr = np.asarray(residual_mem)[gids, :, ni].astype(np.float32, copy=False)
    return ctx, rr


def utility_targets(need_norm, rr_norm, candidate_mask, cfg):
    """Best-alpha relative utility used by the v0.7.1 Utility-A teacher.

    This is diagnostic-only when `need_norm` is the current-query true future
    residual. It is never an inference input.
    """
    c = cfg['predictive_need_probe']['utility_model']
    n = need_norm.permute(0, 2, 1)[:, :, None, :]
    base = torch.abs(n)
    best = torch.full_like(rr_norm, -1e9)
    for a in c['alpha_grid']:
        gain = base - torch.abs(n - float(a) * rr_norm)
        best = torch.maximum(best, gain)
    rel = best / (base + float(c['gain_floor']))
    rel = torch.clamp(rel, -float(c['utility_clip']), float(c['utility_clip']))
    return rel.masked_fill(~candidate_mask[:, :, :, None], -float(c['utility_clip']))


def load_latest(exp_root, variant, subgraph, stage='development'):
    p = Path(exp_root) / variant / subgraph / stage / 'LATEST.json'
    if not p.exists():
        raise RuntimeError(f'{variant} LATEST.json missing: {p}')
    j = json.loads(p.read_text())
    rd = Path(j['run_dir'])
    if not rd.exists():
        raise RuntimeError(f'{variant} latest run_dir missing: {rd}')
    return j, rd


def instantiate_utility_a(checkpoint, device):
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    if ck.get('model_type') != 'support_utility_probe_v071' or ck.get('mode') != 'A':
        raise RuntimeError(
            f'Utility-A checkpoint contract mismatch: model_type={ck.get("model_type")} mode={ck.get("mode")}'
        )
    model = SupportUtilityProbeNet(**ck['model_args']).to(device)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model, ck


def utility_a_batch(
    model,
    node_q,
    node_mem,
    residual_mem,
    base_raw,
    y_raw,
    valid,
    src_ids,
    top4_ids,
    top4_conf,
    top4_mask,
    mean,
    std,
    history,
    horizon,
    cfg,
    device,
):
    """Score M3 Top-4 using the frozen v0.7.1 Utility-A model.

    Returns learned scores and future-grounded utility targets. Current-query
    future is used only to form diagnostic targets, never model inputs.
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
    y = torch.as_tensor(np.array(y_raw[src_ids], copy=True), device=device, dtype=torch.float32)
    qbn = (b - float(mean)) / float(std)
    sbase = cctx[..., 4 * int(history) : 4 * int(history) + int(horizon)]
    need_zero = torch.zeros_like(b)
    true_need = (y - b) / float(std)
    score = model(qctx, cctx, qs, ss, rr, qbn, sbase, conf, need_zero)
    target = utility_targets(true_need, rr, cm, cfg)
    vm = torch.as_tensor(np.array(valid[src_ids], copy=True), device=device, dtype=torch.bool)
    return score, target, rr_raw, cm, vm


def choose_support(score, candidate_mask):
    """Choose one Top-4 support independently per node x horizon."""
    use = candidate_mask[:, :, :, None].expand_as(score)
    masked = score.masked_fill(~use, -1e6)
    pi = masked.argmax(2)  # [B,N,P]
    max_score = torch.gather(score, 2, pi[:, :, None, :]).squeeze(2)
    available = candidate_mask.any(2)[:, :, None].expand_as(max_score)
    max_score = max_score.masked_fill(~available, -1e6)
    return pi, max_score


def gather_chosen_residual(rr_raw, pi):
    """rr_raw [B,N,L,P], pi [B,N,P] -> [B,P,N]."""
    x = torch.gather(rr_raw, 2, pi[:, :, None, :]).squeeze(2)  # [B,N,P]
    return x.permute(0, 2, 1).contiguous()


def quantile_threshold(scores, valid_mask, keep_fraction):
    s = np.asarray(scores, np.float64)
    m = np.asarray(valid_mask, bool) & np.isfinite(s)
    if not np.any(m):
        return float('inf')
    f = float(keep_fraction)
    if f >= 1.0:
        return float('-inf')
    if f <= 0.0:
        return float('inf')
    return float(np.quantile(s[m], 1.0 - f))


def mae_views(pred, truth, valid, direct, support):
    pred = np.asarray(pred, np.float64)
    truth = np.asarray(truth, np.float64)
    valid = np.asarray(valid, bool)
    direct3 = np.asarray(direct, bool)[:, None, :]
    support3 = np.asarray(support, bool)[:, None, :]
    err = np.abs(pred - truth)
    out = {}
    for name, mask in {
        'overall': valid,
        'masked': valid & direct3,
        'support': valid & support3,
    }.items():
        out[name] = float(err[mask].mean()) if np.any(mask) else None
    return out


def relative_gains(base_mae, new_mae):
    out = {}
    for k, b in base_mae.items():
        n = new_mae.get(k)
        out[k] = None if b in (None, 0.0) or n is None else float((b - n) / b)
    return out


def activation_diagnostics(base, truth, corrected, gate, valid, direct, support):
    base = np.asarray(base, np.float64)
    truth = np.asarray(truth, np.float64)
    corrected = np.asarray(corrected, np.float64)
    gate = np.asarray(gate, bool)
    valid = np.asarray(valid, bool)
    direct3 = np.asarray(direct, bool)[:, None, :]
    support3 = np.asarray(support, bool)[:, None, :]
    eb = np.abs(base - truth)
    ec = np.abs(corrected - truth)
    out = {}
    for name, view in {
        'overall': valid,
        'masked': valid & direct3,
        'support': valid & support3,
    }.items():
        act = view & gate
        denom = int(view.sum())
        n = int(act.sum())
        if n == 0:
            out[name] = {
                'view_count': denom, 'activated_count': 0, 'activation_rate': 0.0,
                'harmful_fraction': None, 'beneficial_fraction': None,
                'mean_abs_error_change': None,
            }
        else:
            delta = ec[act] - eb[act]
            out[name] = {
                'view_count': denom,
                'activated_count': n,
                'activation_rate': float(n / max(denom, 1)),
                'harmful_fraction': float(np.mean(delta > 1e-9)),
                'beneficial_fraction': float(np.mean(delta < -1e-9)),
                'mean_abs_error_change': float(np.mean(delta)),
            }
    return out


@dataclass
class RankAccumulator:
    top1_ok: int = 0
    top1_n: int = 0
    rank_ok: int = 0
    rank_n: int = 0
    selected_beneficial: int = 0
    selected_harmful: int = 0
    selected_strong_harm: int = 0
    selected_n: int = 0
    oracle_available: int = 0
    chosen_utility_sum: float = 0.0

    def update(self, score, target, candidate_mask, valid_cell, beneficial_threshold):
        """score/target [B,N,L,P], valid_cell [B,N,P]."""
        use = candidate_mask[:, :, :, None] & valid_cell[:, :, None, :]
        ms = score.masked_fill(~use, -1e6)
        mt = target.masked_fill(~use, -1e6)
        pi = ms.argmax(2)
        ti = mt.argmax(2)
        ok = (use.sum(2) > 0) & valid_cell
        self.top1_ok += int(((pi == ti) & ok).sum().cpu())
        self.top1_n += int(ok.sum().cpu())

        chosen = torch.gather(target, 2, pi[:, :, None, :]).squeeze(2)
        if torch.any(ok):
            cv = chosen[ok]
            self.selected_beneficial += int((cv > float(beneficial_threshold)).sum().cpu())
            self.selected_harmful += int((cv < 0.0).sum().cpu())
            self.selected_strong_harm += int((cv < -float(beneficial_threshold)).sum().cpu())
            self.selected_n += int(cv.numel())
            self.chosen_utility_sum += float(cv.sum().cpu())
            best = mt.max(2).values
            self.oracle_available += int(((best > float(beneficial_threshold)) & ok).sum().cpu())

        hi = mt.argmax(2)
        lo = target.masked_fill(~use, 1e6).argmin(2)
        sp = torch.gather(score, 2, hi[:, :, None, :]).squeeze(2)
        sn = torch.gather(score, 2, lo[:, :, None, :]).squeeze(2)
        tp = torch.gather(target, 2, hi[:, :, None, :]).squeeze(2)
        tn = torch.gather(target, 2, lo[:, :, None, :]).squeeze(2)
        enough = (use.sum(2) >= 2) & ((tp - tn) > 0.02) & valid_cell
        self.rank_ok += int(((sp > sn) & enough).sum().cpu())
        self.rank_n += int(enough.sum().cpu())

    def summary(self):
        return {
            'cell_count': int(self.selected_n),
            'pairwise_rank_count': int(self.rank_n),
            'pairwise_rank_accuracy': float(self.rank_ok / max(self.rank_n, 1)),
            'top1_agreement': float(self.top1_ok / max(self.top1_n, 1)),
            'selected_beneficial_rate': float(self.selected_beneficial / max(self.selected_n, 1)),
            'selected_harmful_rate': float(self.selected_harmful / max(self.selected_n, 1)),
            'selected_strong_harm_rate': float(self.selected_strong_harm / max(self.selected_n, 1)),
            'oracle_any_beneficial_rate': float(self.oracle_available / max(self.selected_n, 1)),
            'beneficial_capture_ratio': float(self.selected_beneficial / max(self.oracle_available, 1)),
            'mean_selected_relative_utility': float(self.chosen_utility_sum / max(self.selected_n, 1)),
        }
