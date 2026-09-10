#!/usr/bin/env python3
"""Fit/apply Validation-outcome-calibrated Support Opportunity (v2).

The final SO is the percentile of predicted *safe global relative headroom*.
Only future-free structural features are model inputs. Realised future utility is
used on Validation only to fit the calibrator; Test only applies the frozen
Validation model and score distribution.

Validation scores are out-of-fold (OOF), so Val grouping is not based on
in-sample fitted predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from supportbench.config import load_config
from supportbench.utils import ensure_dir, log, write_json


DEFAULT_FEATURES = [
    "predictive_context_percentile",
    "support_scope",
    "memory_coverage_percentile",
    "context_scope_interaction",
]
DEFAULT_ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
TARGET_KEY = "oracle_safe_global_relative_gain"


def compress_reference(arr: np.ndarray) -> Dict[str, object]:
    x = np.asarray(arr, float)
    x = x[np.isfinite(x)]
    values, counts = np.unique(x, return_counts=True)
    return {
        "values": [float(v) for v in values],
        "counts": [int(c) for c in counts],
        "count": int(counts.sum()),
    }


def midrank_ecdf(arr: np.ndarray, ref: Dict[str, object]) -> np.ndarray:
    x = np.asarray(arr, float)
    rv = np.asarray(ref.get("values", []), float)
    rc = np.asarray(ref.get("counts", []), np.int64)
    n = int(ref.get("count", int(rc.sum())))
    if n <= 0 or len(rv) == 0:
        return np.full(len(x), 0.5, float)
    cum = np.concatenate([[0], np.cumsum(rc)])
    li = np.searchsorted(rv, x, side="left")
    ri = np.searchsorted(rv, x, side="right")
    less = cum[np.clip(li, 0, len(rv))].astype(float)
    equal = np.zeros(len(x), float)
    exact = (li < len(rv)) & (ri > li)
    if np.any(exact):
        equal[exact] = rc[li[exact]].astype(float)
    out = (less + 0.5 * equal) / float(n)
    out[x < rv[0]] = 0.0
    out[x > rv[-1]] = 1.0
    out[~np.isfinite(x)] = 0.0
    return np.clip(out, 0.0, 1.0)


def load_target(utility_path: str) -> pd.DataFrame:
    u = np.load(utility_path, allow_pickle=False)
    if TARGET_KEY not in u.files:
        raise KeyError(
            f"{TARGET_KEY} missing from utility. Rebuild utility with patched build_support_utility.py"
        )
    return pd.DataFrame(
        {
            "view_id": u["view_id"].astype(str),
            TARGET_KEY: np.asarray(u[TARGET_KEY], float),
            "has_safe_support": np.asarray(u["has_safe_support"], int),
        }
    )


def prepare_xy(features: pd.DataFrame, utility_path: str, feature_names: List[str]):
    u = load_target(utility_path)
    f = features.copy()
    f["view_id"] = f["view_id"].astype(str)
    df = f.merge(u, on="view_id", how="inner", validate="one_to_one")
    if len(df) != len(f):
        raise ValueError(f"Feature/utility join lost rows: features={len(f)}, joined={len(df)}")
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise KeyError(f"Missing features: {missing}")
    X = df[feature_names].to_numpy(float)
    y = np.clip(df[TARGET_KEY].to_numpy(float), 0.0, None)
    if not np.all(np.isfinite(X)):
        raise ValueError("Non-finite future-free features found")
    if not np.all(np.isfinite(y)):
        raise ValueError("Non-finite opportunity target found")
    return df, X, y


def fit_one(Xtr, ytr, Xva, alpha):
    scaler = StandardScaler().fit(Xtr)
    model = Ridge(alpha=float(alpha)).fit(scaler.transform(Xtr), ytr)
    return model.predict(scaler.transform(Xva)), scaler, model


def choose_alpha_oof(X: np.ndarray, y: np.ndarray, alphas: List[float], folds: int, seed: int):
    n = len(y)
    k = min(max(2, int(folds)), n)
    splitter = KFold(n_splits=k, shuffle=True, random_state=int(seed))
    rows = []
    for alpha in alphas:
        pred = np.zeros(n, float)
        for tr, va in splitter.split(X):
            p, _, _ = fit_one(X[tr], y[tr], X[va], alpha)
            pred[va] = p
        pred = np.clip(pred, 0.0, None)
        rows.append(
            {
                "alpha": float(alpha),
                "mae": float(np.mean(np.abs(pred - y))),
                "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
            }
        )
    best = min(rows, key=lambda r: (r["mae"], r["rmse"]))
    return float(best["alpha"]), rows


def oof_predictions(X: np.ndarray, y: np.ndarray, alpha: float, folds: int, seed: int):
    n = len(y)
    k = min(max(2, int(folds)), n)
    pred = np.zeros(n, float)
    splitter = KFold(n_splits=k, shuffle=True, random_state=int(seed))
    for tr, va in splitter.split(X):
        p, _, _ = fit_one(X[tr], y[tr], X[va], alpha)
        pred[va] = p
    return np.clip(pred, 0.0, None)


def fit_full(X: np.ndarray, y: np.ndarray, alpha: float):
    scaler = StandardScaler().fit(X)
    model = Ridge(alpha=float(alpha)).fit(scaler.transform(X), y)
    return scaler, model


def predict_frozen(X: np.ndarray, cal: Dict[str, object]) -> np.ndarray:
    mean = np.asarray(cal["scaler_mean"], float)
    scale = np.asarray(cal["scaler_scale"], float)
    coef = np.asarray(cal["ridge_coef"], float)
    intercept = float(cal["ridge_intercept"])
    z = (X - mean[None, :]) / np.where(scale[None, :] > 1e-12, scale[None, :], 1.0)
    return np.clip(z @ coef + intercept, 0.0, None)


def family_composition(df: pd.DataFrame) -> Dict[str, object]:
    all_counts = df["family"].value_counts().sort_index()
    high = df[df["opportunity_group"] == "high"]
    high_counts = high["family"].value_counts().reindex(all_counts.index, fill_value=0)
    total_high = max(int(high_counts.sum()), 1)
    return {
        "all_counts": {str(k): int(v) for k, v in all_counts.items()},
        "high_counts": {str(k): int(v) for k, v in high_counts.items()},
        "high_composition": {str(k): float(v / total_high) for k, v in high_counts.items()},
        "within_family_high_rate": {
            str(k): float(high_counts[k] / max(int(all_counts[k]), 1)) for k in all_counts.index
        },
    }


def apply_groups(df: pd.DataFrame, raw_pred: np.ndarray, cal: Dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    out["predicted_safe_global_relative_gain"] = raw_pred
    out["support_opportunity"] = midrank_ecdf(raw_pred, cal["validation_score_reference"])
    low = float(cal["opportunity_low_threshold"])
    high = float(cal["opportunity_high_threshold"])
    out["opportunity_group"] = np.where(
        out["support_opportunity"] >= high,
        "high",
        np.where(out["support_opportunity"] <= low, "low", "medium"),
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--features", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fit-calibrator", action="store_true")
    g.add_argument("--calibrator", default=None)
    ap.add_argument("--utility", default=None, help="Validation utility; required when fitting")
    ap.add_argument("--feature-names", nargs="*", default=None)
    ap.add_argument("--cv-folds", type=int, default=None)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--alphas", nargs="*", type=float, default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    challenge = cfg["tracks"]["support_challenge"]
    opcfg = cfg.get("support_opportunity_v2", {})
    feature_names = args.feature_names or list(opcfg.get("feature_names", DEFAULT_FEATURES))
    folds = int(args.cv_folds or opcfg.get("cv_folds", 5))
    alphas = list(map(float, args.alphas or opcfg.get("ridge_alphas", DEFAULT_ALPHAS)))
    f = pd.read_csv(args.features)

    if args.fit_calibrator:
        if not args.utility:
            raise ValueError("--utility is required with --fit-calibrator")
        splits = set(f["split"].astype(str))
        if splits != {"val"}:
            raise ValueError(f"Final SO calibrator must fit on Validation only; got {sorted(splits)}")
        joined, X, y = prepare_xy(f, args.utility, feature_names)
        alpha, alpha_table = choose_alpha_oof(X, y, alphas, folds, args.seed)
        oof = oof_predictions(X, y, alpha, folds, args.seed)
        scaler, model = fit_full(X, y, alpha)
        score_ref = compress_reference(oof)

        provisional = {
            "validation_score_reference": score_ref,
            "opportunity_low_threshold": 0.0,
            "opportunity_high_threshold": 1.0,
        }
        val_pct = midrank_ecdf(oof, score_ref)
        low_q = float(challenge.get("low_quantile", 0.30))
        high_q = float(challenge.get("high_quantile", 0.70))
        low_thr = float(np.quantile(val_pct, low_q))
        high_thr = float(np.quantile(val_pct, high_q))

        rho = (
            float(spearmanr(oof, y).statistic)
            if np.std(oof) > 1e-12 and np.std(y) > 1e-12
            else None
        )
        cal = {
            "schema": "support_opportunity_outcome_calibrator_v2",
            "fitted_split": "val",
            "future_features_only_at_inference": True,
            "validation_outcomes_used_for_fitting": True,
            "test_future_used": False,
            "target_key": TARGET_KEY,
            "target_semantics": "best globally non-harmful gain among support-beneficial candidates, divided by reference global MAE",
            "feature_names": feature_names,
            "cv_folds": int(folds),
            "seed": int(args.seed),
            "ridge_alpha": float(alpha),
            "ridge_alpha_selection": alpha_table,
            "scaler_mean": [float(x) for x in scaler.mean_],
            "scaler_scale": [float(x) for x in scaler.scale_],
            "ridge_coef": [float(x) for x in model.coef_],
            "ridge_intercept": float(model.intercept_),
            "validation_oof_spearman": rho,
            "validation_target_mean": float(np.mean(y)),
            "validation_oof_prediction_mean": float(np.mean(oof)),
            "validation_score_reference": score_ref,
            "low_quantile": low_q,
            "high_quantile": high_q,
            "opportunity_low_threshold": low_thr,
            "opportunity_high_threshold": high_thr,
        }
        out_df = apply_groups(joined.drop(columns=[TARGET_KEY, "has_safe_support"]), oof, cal)
        out_df[TARGET_KEY] = y
        out_df["has_safe_support"] = joined["has_safe_support"].to_numpy(int)
        cal_path = Path(args.output).with_suffix(".calibrator.json")
        write_json(cal_path, cal)
        log(
            f"fitted Validation SO-v2 calibrator; alpha={alpha:g}; OOF spearman={rho}; "
            f"low/high={low_thr:.4f}/{high_thr:.4f}; wrote {cal_path}"
        )
    else:
        cal = json.loads(Path(args.calibrator).read_text(encoding="utf-8"))
        missing = [c for c in cal["feature_names"] if c not in f.columns]
        if missing:
            raise KeyError(f"Missing frozen-calibrator features: {missing}")
        X = f[cal["feature_names"]].to_numpy(float)
        raw = predict_frozen(X, cal)
        out_df = apply_groups(f, raw, cal)

    out = Path(args.output)
    ensure_dir(out.parent)
    out_df.to_csv(out, index=False)
    comp = family_composition(out_df)
    comp_path = out.with_name(out.stem + "__family_composition.json")
    write_json(comp_path, comp)

    counts = out_df["opportunity_group"].value_counts().to_dict()
    max_family = max(comp["high_composition"].values(), default=0.0)
    log(
        f"wrote {out}; groups={counts}; max High-SO family fraction={max_family:.3f}; "
        f"family composition is diagnostic only and never rebalanced"
    )


if __name__ == "__main__":
    main()
