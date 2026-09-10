#!/usr/bin/env python3
"""Build future-free structural support-opportunity features.

This script deliberately does NOT define the final Support Opportunity score.
It produces only inference-time observable features and Validation-fitted
calibration objects.  Test applies the frozen Validation calibrator.

Key design choices (v2):
  * Chapter-3 predictive-context loss is the network mean of per-target loss
    and is converted to a tie-aware family-wise empirical percentile;
  * support-dependent scope is kept as a separate feature;
  * memory coverage is a Validation-fitted empirical percentile of top-1
    retrieval score;
  * retrieval entropy/margins are retained as retrieval-difficulty diagnostics
    and are not mixed into the opportunity score.

No query future, forecast error, or realised support gain is accessed here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from supportbench.config import load_config
from supportbench.dataset import SupportBench
from supportbench.utils import ensure_dir, log, write_json


def normalized_entropy(scores: np.ndarray, temperature: float) -> float:
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) <= 1:
        return 0.0
    z = (s - np.max(s)) / max(float(temperature), 1e-6)
    p = np.exp(z)
    p = p / max(float(p.sum()), 1e-12)
    h = -float(np.sum(p * np.log(np.maximum(p, 1e-12))))
    return h / max(float(np.log(len(p))), 1e-12)


def score_diagnostics(scores: np.ndarray, temperature: float) -> Dict[str, float]:
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if not len(s):
        return {
            "top1_retrieval_score": -1.0,
            "retrieval_ambiguity": 0.0,
            "retrieval_top1_margin": 0.0,
            "retrieval_score_std": 0.0,
        }
    ss = np.sort(s)[::-1]
    top1 = float(ss[0])
    margin = float(ss[0] - ss[1]) if len(ss) > 1 else 0.0
    return {
        "top1_retrieval_score": top1,
        "retrieval_ambiguity": normalized_entropy(s, temperature),
        "retrieval_top1_margin": margin,
        "retrieval_score_std": float(np.std(s)),
    }


def raw_rows(b: SupportBench, pool, temperature: float) -> pd.DataFrame:
    rows = []
    for q, vid0 in enumerate(pool["view_id"]):
        vid = str(vid0)
        sample = b.sample(vid)
        cm = b.predictive_context_metrics(sample)
        valid = np.flatnonzero(
            (pool["candidate_episode_id"][q] >= 0)
            & np.isfinite(pool["retrieval_score"][q])
        )
        scores = pool["retrieval_score"][q, valid] if len(valid) else np.asarray([])
        sd = score_diagnostics(scores, temperature)
        rows.append(
            {
                "view_id": vid,
                "view_group_id": str(sample.view_group_id),
                "split": str(sample.split),
                "window_start": int(sample.window_start),
                "family": str(sample.view_spec["family"]),
                "severity": float(sample.view_spec["severity"]),
                "distribution": str(sample.view_spec.get("distribution", "id")),
                "predictive_context_loss": float(cm["network_mean_context_loss"]),
                "network_mean_context_loss": float(cm["network_mean_context_loss"]),
                "legacy_context_loss_quantile": float(cm["context_loss_quantile_value"]),
                "support_dependent_fraction": float(cm["support_dependent_fraction"]),
                "valid_candidate_count": int(len(valid)),
                **sd,
            }
        )
    return pd.DataFrame(rows)


def _compress_reference(arr: Iterable[float]) -> Dict[str, object]:
    x = np.asarray(list(arr), dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"values": [], "counts": [], "count": 0, "constant": True}
    values, counts = np.unique(x, return_counts=True)
    return {
        "values": [float(v) for v in values],
        "counts": [int(c) for c in counts],
        "count": int(counts.sum()),
        "constant": bool(len(values) <= 1),
    }


def _midrank_ecdf(values: np.ndarray, ref: Dict[str, object]) -> np.ndarray:
    """Tie-aware Validation empirical percentile.

    For x exactly tied with a Validation mass, assign the midpoint of that mass:
      (count(<x) + 0.5*count(=x)) / N.
    This avoids mapping an 80% upper tie to 1.0 for every tied sample.
    """
    x = np.asarray(values, dtype=float)
    rv = np.asarray(ref.get("values", []), dtype=float)
    rc = np.asarray(ref.get("counts", []), dtype=np.int64)
    n = int(ref.get("count", int(rc.sum())))
    if n <= 0 or len(rv) == 0:
        return np.full(len(x), 0.5, dtype=np.float64)

    cum = np.concatenate([[0], np.cumsum(rc)])
    li = np.searchsorted(rv, x, side="left")
    ri = np.searchsorted(rv, x, side="right")
    less = cum[np.clip(li, 0, len(rv))].astype(np.float64)

    equal = np.zeros(len(x), dtype=np.float64)
    exact = (li < len(rv)) & (ri > li)
    if np.any(exact):
        equal[exact] = rc[li[exact]].astype(np.float64)

    out = (less + 0.5 * equal) / float(n)
    out[x < rv[0]] = 0.0
    out[x > rv[-1]] = 1.0
    out[~np.isfinite(x)] = 0.0
    return np.clip(out, 0.0, 1.0)


def fit_calibrator(df: pd.DataFrame) -> Dict[str, object]:
    context_refs: Dict[str, object] = {
        "__global__": _compress_reference(df["predictive_context_loss"].to_numpy(float))
    }
    for family, g in df.groupby("family", sort=True):
        context_refs[str(family)] = _compress_reference(
            g["predictive_context_loss"].to_numpy(float)
        )
    return {
        "schema": "support_opportunity_features_v2",
        "fitted_split": str(df["split"].iloc[0]) if len(df) else "unknown",
        "future_free": True,
        "context_calibration": "validation_family_midrank_ecdf",
        "memory_coverage_calibration": "validation_global_midrank_ecdf",
        "context_loss_family_references": context_refs,
        "top1_retrieval_reference": _compress_reference(
            df["top1_retrieval_score"].to_numpy(float)
        ),
        "context_loss_metric": "network_mean_context_loss = mean(per_target_context_loss)",
        "legacy_context_metric": "legacy_context_loss_quantile is retained only for v1.2.2 diagnostics",
        "notes": {
            "retrieval_ambiguity": "diagnostic retrieval difficulty only; excluded from final opportunity calibration by default",
            "support_dependent_fraction": "future-free scope of targets whose Train-defined predictive context is disrupted",
        },
    }


def apply_calibrator(df: pd.DataFrame, cal: Dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    refs = cal["context_loss_family_references"]
    ctx = np.zeros(len(out), dtype=np.float64)
    for family, idx in out.groupby("family").groups.items():
        ref = refs.get(str(family), refs.get("__global__"))
        if ref is None:
            raise KeyError(f"No Validation context reference for family={family!r}")
        arr = out.loc[idx, "predictive_context_loss"].to_numpy(float)
        ctx[out.index.get_indexer(idx)] = _midrank_ecdf(arr, ref)

    out["predictive_context_percentile"] = ctx
    out["support_scope"] = np.clip(
        out["support_dependent_fraction"].to_numpy(float), 0.0, 1.0
    )
    out["memory_coverage_percentile"] = _midrank_ecdf(
        out["top1_retrieval_score"].to_numpy(float),
        cal["top1_retrieval_reference"],
    )
    # Interpretable interaction: disruption intensity is only useful when it
    # affects some forecast targets.  This remains a feature, not a fixed score.
    out["context_scope_interaction"] = (
        out["predictive_context_percentile"].to_numpy(float)
        * np.sqrt(out["support_scope"].to_numpy(float))
    )
    # Difficulty is intentionally separate from opportunity.
    out["retrieval_difficulty"] = out["retrieval_ambiguity"].astype(float)
    return out


def family_feature_summary(df: pd.DataFrame) -> Dict[str, object]:
    cols = [
        "predictive_context_loss",
        "predictive_context_percentile",
        "support_scope",
        "memory_coverage_percentile",
        "retrieval_difficulty",
        "retrieval_top1_margin",
    ]
    result: Dict[str, object] = {}
    for family, g in df.groupby("family", sort=True):
        result[str(family)] = {
            c: {
                "mean": float(np.nanmean(g[c].to_numpy(float))),
                "std": float(np.nanstd(g[c].to_numpy(float))),
                "q10": float(np.nanquantile(g[c].to_numpy(float), 0.10)),
                "q50": float(np.nanquantile(g[c].to_numpy(float), 0.50)),
                "q90": float(np.nanquantile(g[c].to_numpy(float), 0.90)),
            }
            for c in cols
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--subgraph", default="TraffiDent_Net_06")
    ap.add_argument("--candidate-pool", required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--fit-calibrator", action="store_true")
    group.add_argument("--calibrator", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    b = SupportBench(args.subgraph, config=args.config)
    pool = np.load(args.candidate_pool, allow_pickle=False)
    challenge = cfg["tracks"]["support_challenge"]
    temperature = float(challenge.get("softmax_temperature", 0.10))

    df = raw_rows(b, pool, temperature)
    if args.fit_calibrator:
        splits = set(df["split"].astype(str))
        if splits != {"val"}:
            raise ValueError(
                f"Feature calibrator must be fitted on Validation only; got {sorted(splits)}"
            )
        cal = fit_calibrator(df)
        cal_path = Path(args.output).with_suffix(".calibrator.json")
        write_json(cal_path, cal)
        log(f"wrote Validation structural-feature calibrator {cal_path}")
    else:
        cal = json.loads(Path(args.calibrator).read_text(encoding="utf-8"))

    out_df = apply_calibrator(df, cal)
    out = Path(args.output)
    ensure_dir(out.parent)
    out_df.to_csv(out, index=False)
    summary_path = out.with_name(out.stem + "__family_features.json")
    write_json(summary_path, family_feature_summary(out_df))

    log(
        f"wrote {out}; queries={len(out_df)}; "
        f"context_pct mean/std={out_df.predictive_context_percentile.mean():.4f}/"
        f"{out_df.predictive_context_percentile.std():.4f}; "
        f"scope mean={out_df.support_scope.mean():.4f}; "
        f"memory_pct mean={out_df.memory_coverage_percentile.mean():.4f}; "
        f"retrieval_difficulty mean/std={out_df.retrieval_difficulty.mean():.4f}/"
        f"{out_df.retrieval_difficulty.std():.4f}; summary={summary_path}"
    )


if __name__ == "__main__":
    main()
