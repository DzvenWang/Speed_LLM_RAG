#!/usr/bin/env python3
"""Compute backbone-conditioned predictive-support utility and query headroom.

v2 keeps candidate-level multi-granularity utility from v1.2.2 and additionally
stores query-level oracle/similarity-top1 summaries needed to calibrate Support
Opportunity on Validation without re-reading futures later.

Future labels are used only for offline utility supervision/diagnostics. They are
never inference-time inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from supportbench.config import load_config
from supportbench.dataset import SupportBench
from supportbench.utils import ensure_dir, log


def masked_mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    aa = np.asarray(a, np.float32)
    bb = np.asarray(b, np.float32)
    mm = np.asarray(mask, bool) & np.isfinite(aa) & np.isfinite(bb)
    return float(np.mean(np.abs(aa[mm] - bb[mm]))) if np.any(mm) else float("nan")


def resolve_alpha(args, cfg):
    if args.alpha_config:
        obj = json.loads(Path(args.alpha_config).read_text())
        return float(obj["selected_alpha"]), str(Path(args.alpha_config).resolve())
    if args.alpha is not None:
        return float(args.alpha), "explicit_cli"
    if args.mode == "raw_future":
        return 0.20, "raw_future_default_ablation"
    raise ValueError(
        "Residual utility requires --alpha-config selected on Validation (preferred) or explicit --alpha."
    )


def metric_masks(b: SupportBench, sample, direct_threshold: float):
    valid = np.asarray(sample.raw_valid_future, bool)
    direct_fraction = np.asarray(sample.artificial_mask, bool).mean(axis=0)
    direct = direct_fraction > float(direct_threshold)
    cm = b.predictive_context_metrics(sample)
    support = np.asarray(cm["support_dependent_target_mask"], bool)
    return {
        "all": valid,
        "masked": valid & direct[None, :],
        "support_dependent": valid & support[None, :],
    }, cm, int(direct.sum()), int(support.sum())


def safe_div(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den <= 1e-12:
        return 0.0
    return float(num / den)


def query_headroom(
    ids: np.ndarray,
    retrieval_score: np.ndarray,
    gain_support: np.ndarray,
    gain_all: np.ndarray,
    safe_support: np.ndarray,
    base_support: np.ndarray,
    base_all: np.ndarray,
):
    """Return per-query realised headroom summaries.

    Opportunity target used downstream:
      oracle_safe_global_relative_gain
    = best global MAE gain among candidates that are beneficial on the
      support-dependent scope and globally non-harmful, divided by reference
      global MAE.  This aligns SO with "is safe enhancement worth invoking?".

    The reasoner can still use gain_support_dependent as its primary
    candidate-level supervision target.
    """
    qn = ids.shape[0]
    names = [
        "has_safe_support",
        "oracle_support_gain",
        "oracle_safe_support_gain",
        "oracle_safe_global_gain",
        "oracle_safe_support_relative_gain",
        "oracle_safe_global_relative_gain",
        "similarity_top1_support_gain",
        "similarity_top1_global_gain",
        "similarity_top1_safe",
    ]
    out = {n: np.zeros(qn, np.float32) for n in names}
    out["has_safe_support"] = np.zeros(qn, np.uint8)
    out["similarity_top1_safe"] = np.zeros(qn, np.uint8)

    for q in range(qn):
        valid = np.flatnonzero(
            (ids[q] >= 0)
            & np.isfinite(retrieval_score[q])
            & np.isfinite(gain_support[q])
        )
        if not len(valid):
            continue

        out["oracle_support_gain"][q] = max(
            0.0, float(np.nanmax(gain_support[q, valid]))
        )

        safe_idx = valid[safe_support[q, valid].astype(bool)]
        if len(safe_idx):
            out["has_safe_support"][q] = 1
            best_support = float(np.nanmax(gain_support[q, safe_idx]))
            # SO is about final safe pathway value, so choose the safe candidate
            # with the best global gain for the global opportunity target.
            finite_global = safe_idx[np.isfinite(gain_all[q, safe_idx])]
            best_global = (
                max(0.0, float(np.nanmax(gain_all[q, finite_global])))
                if len(finite_global)
                else 0.0
            )
            out["oracle_safe_support_gain"][q] = max(0.0, best_support)
            out["oracle_safe_global_gain"][q] = best_global
            out["oracle_safe_support_relative_gain"][q] = safe_div(
                max(0.0, best_support), float(base_support[q])
            )
            out["oracle_safe_global_relative_gain"][q] = safe_div(
                best_global, float(base_all[q])
            )

        top = int(valid[np.argmax(retrieval_score[q, valid])])
        out["similarity_top1_support_gain"][q] = float(gain_support[q, top])
        out["similarity_top1_global_gain"][q] = (
            float(gain_all[q, top]) if np.isfinite(gain_all[q, top]) else 0.0
        )
        out["similarity_top1_safe"][q] = np.uint8(bool(safe_support[q, top]))

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--subgraph", default="TraffiDent_Net_06")
    ap.add_argument("--candidate-pool", required=True)
    ap.add_argument("--query-predictions", required=True)
    ap.add_argument("--memory-predictions", default=None)
    ap.add_argument("--mode", choices=["residual", "raw_future"], default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--alpha-config", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    args.mode = args.mode or str(cfg["support_utility"]["primary_mode"])
    alpha, alpha_source = resolve_alpha(args, cfg)
    b = SupportBench(args.subgraph, config=args.config)
    pool = np.load(args.candidate_pool, allow_pickle=False)
    qp = np.load(args.query_predictions, allow_pickle=False)
    pmap = {str(v): i for i, v in enumerate(qp["view_id"])}
    mp = None
    mmap = {}
    if args.mode == "residual":
        if not args.memory_predictions:
            raise ValueError("--memory-predictions required for residual mode")
        mp = np.load(args.memory_predictions, allow_pickle=False)
        mmap = {int(e): i for i, e in enumerate(mp["episode_id"])}

    Q, K = pool["candidate_episode_id"].shape
    scopes = ["all", "masked", "support_dependent"]
    base = {x: np.full(Q, np.nan, np.float32) for x in scopes}
    aug = {x: np.full((Q, K), np.nan, np.float32) for x in scopes}
    gain = {x: np.full((Q, K), np.nan, np.float32) for x in scopes}
    horizons = [h for h in (3, 6, 12) if h <= b.P]
    base_h = {h: np.full(Q, np.nan, np.float32) for h in horizons}
    aug_h = {h: np.full((Q, K), np.nan, np.float32) for h in horizons}
    gain_h = {h: np.full((Q, K), np.nan, np.float32) for h in horizons}
    base_h_support = {h: np.full(Q, np.nan, np.float32) for h in horizons}
    aug_h_support = {h: np.full((Q, K), np.nan, np.float32) for h in horizons}
    gain_h_support = {h: np.full((Q, K), np.nan, np.float32) for h in horizons}
    direct_count = np.zeros(Q, np.int32)
    support_count = np.zeros(Q, np.int32)
    support_frac = np.zeros(Q, np.float32)
    context_loss = np.zeros(Q, np.float32)
    legacy_context_quantile = np.zeros(Q, np.float32)
    direct_thr = float(cfg["support_utility"].get("direct_masked_missing_threshold", 0.0))
    safety_tol = float(cfg["support_utility"].get("global_safety_tolerance_mae", 0.0))

    for q, vid0 in enumerate(pool["view_id"]):
        vid = str(vid0)
        if vid not in pmap:
            raise KeyError(f"{vid} missing from query predictions")
        sample = b.sample(vid)
        y = np.asarray(sample.future_speed, np.float32)
        base_pred = np.asarray(qp["pred"][pmap[vid]], np.float32)
        if base_pred.shape != y.shape:
            raise ValueError(f"{vid}: prediction {base_pred.shape} != target {y.shape}")
        masks, cm, dc, sc = metric_masks(b, sample, direct_thr)
        direct_count[q] = dc
        support_count[q] = sc
        support_frac[q] = float(cm["support_dependent_fraction"])
        context_loss[q] = float(cm["network_mean_context_loss"])
        legacy_context_quantile[q] = float(cm["context_loss_quantile_value"])
        for scope in scopes:
            base[scope][q] = masked_mae(base_pred, y, masks[scope])
        for h in horizons:
            base_h[h][q] = masked_mae(
                base_pred[h - 1 : h], y[h - 1 : h], masks["all"][h - 1 : h]
            )
            base_h_support[h][q] = masked_mae(
                base_pred[h - 1 : h],
                y[h - 1 : h],
                masks["support_dependent"][h - 1 : h],
            )

        for k, eid0 in enumerate(pool["candidate_episode_id"][q]):
            eid = int(eid0)
            if eid < 0:
                continue
            cy = np.asarray(b.memory_episode(eid)["future_speed"], np.float32)
            if args.mode == "raw_future":
                aug_pred = (1.0 - alpha) * base_pred + alpha * cy
            else:
                if eid not in mmap:
                    raise KeyError(f"memory episode {eid} missing prediction")
                cres = cy - np.asarray(mp["pred"][mmap[eid]], np.float32)
                aug_pred = base_pred + alpha * cres

            for scope in scopes:
                ae = masked_mae(aug_pred, y, masks[scope])
                aug[scope][q, k] = ae
                if np.isfinite(base[scope][q]) and np.isfinite(ae):
                    gain[scope][q, k] = base[scope][q] - ae
            for h in horizons:
                ae = masked_mae(
                    aug_pred[h - 1 : h], y[h - 1 : h], masks["all"][h - 1 : h]
                )
                aug_h[h][q, k] = ae
                if np.isfinite(base_h[h][q]) and np.isfinite(ae):
                    gain_h[h][q, k] = base_h[h][q] - ae
                se = masked_mae(
                    aug_pred[h - 1 : h],
                    y[h - 1 : h],
                    masks["support_dependent"][h - 1 : h],
                )
                aug_h_support[h][q, k] = se
                if np.isfinite(base_h_support[h][q]) and np.isfinite(se):
                    gain_h_support[h][q, k] = base_h_support[h][q] - se

    beneficial_all = np.isfinite(gain["all"]) & (gain["all"] > 0)
    beneficial_support = np.isfinite(gain["support_dependent"]) & (
        gain["support_dependent"] > 0
    )
    safe_support = (
        beneficial_support
        & np.isfinite(gain["all"])
        & (gain["all"] >= -float(safety_tol))
    )

    qhead = query_headroom(
        pool["candidate_episode_id"],
        pool["retrieval_score"],
        gain["support_dependent"],
        gain["all"],
        safe_support,
        base["support_dependent"],
        base["all"],
    )

    out = (
        Path(args.output)
        if args.output
        else Path(args.candidate_pool).with_name(
            Path(args.candidate_pool).stem + f"__utility_{args.mode}.npz"
        )
    )
    ensure_dir(out.parent)
    payload = {
        "view_id": pool["view_id"],
        "candidate_episode_id": pool["candidate_episode_id"],
        "retrieval_score": pool["retrieval_score"],
        "base_mae": base["all"],
        "augmented_mae": aug["all"],
        "forecast_gain": gain["all"],
        "base_mae_all": base["all"],
        "augmented_mae_all": aug["all"],
        "gain_all": gain["all"],
        "base_mae_masked": base["masked"],
        "augmented_mae_masked": aug["masked"],
        "gain_masked": gain["masked"],
        "base_mae_support_dependent": base["support_dependent"],
        "augmented_mae_support_dependent": aug["support_dependent"],
        "gain_support_dependent": gain["support_dependent"],
        "beneficial_all": beneficial_all.astype(np.uint8),
        "beneficial_support_dependent": beneficial_support.astype(np.uint8),
        "safe_support": safe_support.astype(np.uint8),
        "directly_masked_target_count": direct_count,
        "support_dependent_target_count": support_count,
        "support_dependent_fraction": support_frac,
        "predictive_context_loss": context_loss,
        "network_mean_context_loss": context_loss,
        "legacy_context_loss_quantile": legacy_context_quantile,
        "mode": np.asarray(args.mode),
        "alpha": np.float32(alpha),
        "alpha_source": np.asarray(alpha_source),
        "primary_gain_key": np.asarray(
            str(cfg["support_utility"].get("primary_gain_key", "gain_support_dependent"))
        ),
        "opportunity_target_key": np.asarray("oracle_safe_global_relative_gain"),
        "global_safety_tolerance_mae": np.float32(safety_tol),
        **qhead,
    }
    for h in horizons:
        payload[f"base_mae_h{h}"] = base_h[h]
        payload[f"augmented_mae_h{h}"] = aug_h[h]
        payload[f"gain_h{h}"] = gain_h[h]
        payload[f"base_mae_h{h}_support_dependent"] = base_h_support[h]
        payload[f"augmented_mae_h{h}_support_dependent"] = aug_h_support[h]
        payload[f"gain_h{h}_support_dependent"] = gain_h_support[h]
    for key in [
        "temporal_score",
        "spatial_score",
        "propagation_score",
        "candidate_source_mask",
    ]:
        if key in pool.files:
            payload[key] = pool[key]
    np.savez_compressed(out, **payload)

    log(
        f"wrote {out}; mode={args.mode}; alpha={alpha:.3f}; "
        f"mean base(all)={np.nanmean(base['all']):.5f}; "
        f"mean base(support)={np.nanmean(base['support_dependent']):.5f}; "
        f"has-safe-support={float(np.mean(qhead['has_safe_support'])):.3f}; "
        f"mean oracle-safe-global-relative-gain="
        f"{float(np.mean(qhead['oracle_safe_global_relative_gain'])):.6f}"
    )


if __name__ == "__main__":
    main()
