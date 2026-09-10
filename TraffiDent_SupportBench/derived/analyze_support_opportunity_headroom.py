#!/usr/bin/env python3
"""Diagnose whether future-free support features predict realised support headroom.

Use on Validation before fitting the final Support Opportunity calibrator.  Test
may be analysed only after the Validation calibrator is frozen, for reporting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from supportbench.utils import write_json


DEFAULT_FEATURES = [
    "predictive_context_percentile",
    "support_scope",
    "memory_coverage_percentile",
    "context_scope_interaction",
    "retrieval_difficulty",
    "retrieval_top1_margin",
]


def finite_spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3 or np.nanstd(x[m]) < 1e-12 or np.nanstd(y[m]) < 1e-12:
        return None
    return float(spearmanr(x[m], y[m]).statistic)


def aucs(x, label):
    x = np.asarray(x, float)
    y = np.asarray(label, int)
    m = np.isfinite(x)
    if int(m.sum()) < 3 or len(np.unique(y[m])) < 2 or np.nanstd(x[m]) < 1e-12:
        return None, None
    return float(roc_auc_score(y[m], x[m])), float(average_precision_score(y[m], x[m]))


def top_fraction_stats(x, target, frac=0.30):
    x = np.asarray(x, float)
    y = np.asarray(target, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if not len(x):
        return {"count": 0, "mean": None, "overall_mean": None, "lift_ratio": None}
    n = max(1, int(np.ceil(frac * len(x))))
    idx = np.argsort(-x)[:n]
    overall = float(np.mean(y))
    top = float(np.mean(y[idx]))
    return {
        "count": int(n),
        "mean": top,
        "overall_mean": overall,
        "lift_ratio": float(top / overall) if abs(overall) > 1e-12 else None,
        "lift_absolute": float(top - overall),
    }


def load_join(features_path: str, utility_path: str):
    f = pd.read_csv(features_path)
    # The fitted Validation SO CSV may already carry realised targets for
    # convenience; drop them here and re-join from the utility NPZ so the
    # diagnostic has a single authoritative source.
    for c in [
        "has_safe_support",
        "oracle_safe_global_relative_gain",
        "oracle_safe_support_relative_gain",
        "similarity_top1_global_gain",
        "similarity_top1_safe",
    ]:
        if c in f.columns:
            f = f.drop(columns=[c])
    u = np.load(utility_path, allow_pickle=False)
    required = [
        "view_id",
        "has_safe_support",
        "oracle_safe_global_relative_gain",
        "oracle_safe_support_relative_gain",
        "similarity_top1_global_gain",
        "similarity_top1_safe",
    ]
    missing = [k for k in required if k not in u.files]
    if missing:
        raise KeyError(
            f"Utility file is missing v2 query-headroom fields: {missing}. "
            "Rebuild it with the patched build_support_utility.py."
        )
    udf = pd.DataFrame({k: u[k] for k in required})
    udf["view_id"] = udf["view_id"].astype(str)
    f["view_id"] = f["view_id"].astype(str)
    out = f.merge(udf, on="view_id", how="inner", validate="one_to_one")
    if len(out) != len(f):
        raise ValueError(f"Feature/utility join lost rows: features={len(f)}, joined={len(out)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--utility", required=True)
    ap.add_argument("--feature-names", nargs="*", default=None)
    ap.add_argument("--top-fraction", type=float, default=0.30)
    ap.add_argument("--output", default=None)
    ap.add_argument("--query-output", default=None)
    args = ap.parse_args()

    df = load_join(args.features, args.utility)
    features = args.feature_names or [c for c in DEFAULT_FEATURES if c in df.columns]
    target = df["oracle_safe_global_relative_gain"].to_numpy(float)
    support_target = df["oracle_safe_support_relative_gain"].to_numpy(float)
    label = df["has_safe_support"].to_numpy(int)

    feature_report = {}
    for name in features:
        x = df[name].to_numpy(float)
        roc, apv = aucs(x, label)
        feature_report[name] = {
            "spearman_oracle_safe_global_relative_gain": finite_spearman(x, target),
            "spearman_oracle_safe_support_relative_gain": finite_spearman(x, support_target),
            "roc_auc_has_safe_support": roc,
            "average_precision_has_safe_support": apv,
            "top_fraction_global_gain": top_fraction_stats(x, target, args.top_fraction),
        }

    family = {}
    for fam, g in df.groupby("family", sort=True):
        family[str(fam)] = {
            "queries": int(len(g)),
            "has_safe_support_rate": float(g["has_safe_support"].mean()),
            "mean_oracle_safe_global_relative_gain": float(
                g["oracle_safe_global_relative_gain"].mean()
            ),
            "mean_oracle_safe_support_relative_gain": float(
                g["oracle_safe_support_relative_gain"].mean()
            ),
            "mean_similarity_top1_global_gain": float(
                g["similarity_top1_global_gain"].mean()
            ),
            "similarity_top1_safe_rate": float(g["similarity_top1_safe"].mean()),
        }

    report = {
        "features": str(Path(args.features).resolve()),
        "utility": str(Path(args.utility).resolve()),
        "queries": int(len(df)),
        "split": sorted(df["split"].astype(str).unique().tolist()),
        "target": "oracle_safe_global_relative_gain",
        "has_safe_support_rate": float(df["has_safe_support"].mean()),
        "mean_oracle_safe_global_relative_gain": float(target.mean()),
        "mean_oracle_safe_support_relative_gain": float(support_target.mean()),
        "mean_similarity_top1_global_gain": float(df["similarity_top1_global_gain"].mean()),
        "similarity_top1_safe_rate": float(df["similarity_top1_safe"].mean()),
        "feature_diagnostics": feature_report,
        "family_outcome_diagnostics": family,
        "interpretation": {
            "opportunity_features": "Prefer features with positive headroom correlation/top-30% lift; do not select weights from family composition alone.",
            "retrieval_difficulty": "Entropy/margins describe candidate-selection difficulty and should not be forced into Support Opportunity unless Validation outcomes show clear incremental utility.",
        },
    }

    print(json.dumps(report, indent=2))
    out = (
        Path(args.output)
        if args.output
        else Path(args.features).with_name(Path(args.features).stem + "__headroom.json")
    )
    write_json(out, report)
    qout = (
        Path(args.query_output)
        if args.query_output
        else out.with_name(out.stem.replace("__headroom", "") + "__headroom_queries.csv")
    )
    df.to_csv(qout, index=False)


if __name__ == "__main__":
    main()
