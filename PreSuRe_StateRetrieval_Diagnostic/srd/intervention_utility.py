from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Dict
import math
import numpy as np
import pandas as pd

try:
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scikit-learn>=1.3 is required for v0.8 intervention-aware utility") from exc

from .utility_selector import (
    add_inference_features,
    expanding_time_folds,
    blocked_folds,
    tune_gate_and_cap,
    apply_decision_rule,
    select_best_per_patch,
    json_safe,
)

LABEL_COLUMNS = {
    "baseline_forecast_mae", "forecast_mae", "forecast_utility", "forecast_gain_pct",
    "patch_completion_mae", "beneficial", "harmful", "relative_utility", "utility_target",
}

V08_MODELS = [
    "logistic_intervention",
    "cost_sensitive_gate",
    "histgb_mean_intervention",
    "histgb_q20_intervention",
    "risk_expected_intervention",
]

PROPOSAL_FEATURES = [
    "candidate_patch_mean", "candidate_patch_std", "candidate_patch_recent_mean",
    "candidate_patch_recent_std", "candidate_patch_delta_abs_mean",
    "candidate_query_mean_gap", "candidate_query_std_gap", "candidate_query_delta_gap",
    "candidate_neighbor_observed_mae",
    "affine_scale", "affine_shift", "affine_scale_deviation", "affine_overlap_points",
    "affine_at_clip", "proposed_fill_count", "proposal_fill_mean", "proposal_fill_std",
    "proposal_abs_change_mean", "proposal_abs_change_max", "proposal_change_rms",
    "proposal_change_l2", "proposal_anchor_jump_mean", "proposal_anchor_jump_max",
    "proposal_patch_after_mean", "proposal_patch_after_std", "proposal_obs_mean_gap",
    "proposal_neighbor_gap_mean", "proposal_neighbor_gap_max", "proposal_neighbor_support_fraction",
]

DEEP_FEATURES = [
    "prediction_shift_mae", "prediction_shift_rmse", "prediction_shift_max", "prediction_shift_std",
    "prediction_shift_signed_mean", "prediction_patch_shift_mae", "prediction_outside_patch_shift_mae",
    "prediction_spill_ratio", "latent_global_mae", "latent_global_rmse", "latent_global_cosine",
    "latent_patch_mae", "latent_patch_rmse", "latent_patch_cosine", "latent_outside_patch_mae",
    "latent_spill_ratio",
]


def add_v08_features(df: pd.DataFrame) -> pd.DataFrame:
    x = add_inference_features(df)
    for c in PROPOSAL_FEATURES + DEEP_FEATURES:
        if c not in x:
            x[c] = np.nan
    # Stable nonlinear risk indicators. All are inference-available.
    x["proposal_change_per_fill"] = x["proposal_change_l2"].astype(float) / np.sqrt(
        np.maximum(x["proposed_fill_count"].astype(float), 1.0)
    )
    x["anchor_jump_to_patch_std"] = x["proposal_anchor_jump_mean"].astype(float) / np.maximum(
        x["query_patch_obs_std"].astype(float), 1e-3
    )
    x["prediction_patch_to_global_shift"] = x["prediction_patch_shift_mae"].astype(float) / np.maximum(
        x["prediction_shift_mae"].astype(float), 1e-6
    )
    x["latent_patch_to_global_shift"] = x["latent_patch_mae"].astype(float) / np.maximum(
        x["latent_global_mae"].astype(float), 1e-6
    )
    x["affine_shift_abs"] = np.abs(x["affine_shift"].astype(float))
    x["proposal_neighbor_gap_log"] = np.log1p(np.maximum(x["proposal_neighbor_gap_mean"].astype(float), 0.0))
    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    return x


def metadata_feature_columns(df: pd.DataFrame) -> List[str]:
    # Reuse v0.7 inference feature surface, but intentionally exclude the proposal/deep columns.
    from .utility_selector import feature_columns as v07_feature_columns
    return [c for c in v07_feature_columns(df) if c not in PROPOSAL_FEATURES and c not in DEEP_FEATURES]


def feature_columns_v08(df: pd.DataFrame, feature_set: str = "full") -> List[str]:
    x = add_v08_features(df)
    meta = metadata_feature_columns(x)
    proposal = [c for c in PROPOSAL_FEATURES if c in x.columns and not x[c].isna().all()]
    deep = [c for c in DEEP_FEATURES if c in x.columns and not x[c].isna().all()]
    derived = [
        "proposal_change_per_fill", "anchor_jump_to_patch_std", "prediction_patch_to_global_shift",
        "latent_patch_to_global_shift", "affine_shift_abs", "proposal_neighbor_gap_log",
    ]
    derived = [c for c in derived if c in x.columns and not x[c].isna().all()]
    fs = str(feature_set)
    if fs == "metadata":
        cols = meta
    elif fs == "proposal":
        cols = meta + proposal + [c for c in derived if not c.startswith("prediction_") and not c.startswith("latent_")]
    elif fs == "deep":
        cols = meta + deep
    elif fs == "full":
        cols = meta + proposal + deep + derived
    else:
        raise KeyError(f"Unknown v0.8 feature_set={feature_set}")
    # Deduplicate while preserving order.
    cols = list(dict.fromkeys(cols))
    bad = [c for c in cols if c in LABEL_COLUMNS or any(k in c for k in ["completion_mae", "beneficial", "harmful"])]
    if bad:
        raise RuntimeError(f"Leakage-prone v0.8 features unexpectedly included: {bad}")
    return cols


def add_utility_targets(df: pd.DataFrame, harmful_margin: float = 0.01) -> pd.DataFrame:
    x = df.copy()
    base = np.maximum(x["baseline_forecast_mae"].to_numpy(np.float64), 1e-6)
    u = x["forecast_utility"].to_numpy(np.float64)
    x["relative_utility"] = u / base
    x["utility_target"] = x["relative_utility"]
    x["beneficial"] = (u > 0.0).astype(np.int8)
    x["harmful"] = (u < -float(harmful_margin)).astype(np.int8)
    return x


def _imputed_histgb_reg(seed: int, *, loss: str = "squared_error", quantile: float | None = None):
    kwargs = dict(
        max_iter=240, max_leaf_nodes=31, max_depth=6, learning_rate=0.04,
        min_samples_leaf=50, l2_regularization=1.5, random_state=int(seed), loss=loss,
    )
    if quantile is not None:
        kwargs["quantile"] = float(quantile)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(**kwargs)),
    ])


def _imputed_histgb_cls(seed: int):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            max_iter=220, max_leaf_nodes=31, max_depth=6, learning_rate=0.04,
            min_samples_leaf=50, l2_regularization=1.5, random_state=int(seed),
        )),
    ])


@dataclass
class V08Model:
    name: str
    task: str
    estimator: object | None = None
    benefit_estimator: object | float | None = None
    harm_estimator: object | float | None = None
    positive_estimator: object | float | None = None
    harmful_magnitude_estimator: object | float | None = None


def make_v08_model(name: str, seed: int) -> V08Model:
    name = str(name)
    if name == "logistic_intervention":
        return V08Model(name, "classification", estimator=Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=900, class_weight="balanced", random_state=int(seed))),
        ]))
    if name == "cost_sensitive_gate":
        return V08Model(name, "cost_sensitive", estimator=_imputed_histgb_cls(seed))
    if name == "histgb_mean_intervention":
        return V08Model(name, "regression", estimator=_imputed_histgb_reg(seed))
    if name == "histgb_q20_intervention":
        return V08Model(name, "regression", estimator=_imputed_histgb_reg(seed, loss="quantile", quantile=0.20))
    if name == "risk_expected_intervention":
        return V08Model(
            name, "risk_expected",
            benefit_estimator=_imputed_histgb_cls(seed),
            harm_estimator=_imputed_histgb_cls(seed + 17),
            positive_estimator=_imputed_histgb_reg(seed + 31),
            harmful_magnitude_estimator=_imputed_histgb_reg(seed + 47),
        )
    raise KeyError(f"Unknown v0.8 model={name}")


def _fit_binary(est, X, y):
    if np.unique(y).size < 2:
        return float(np.mean(y))
    est.fit(X, y)
    return est


def _fit_subset_reg(est, X, y, mask):
    if int(np.sum(mask)) < 30:
        vals = y[mask]
        return float(np.mean(vals)) if vals.size else 0.0
    est.fit(X[mask], y[mask])
    return est


def fit_v08_model(spec: V08Model, train_df: pd.DataFrame, features: List[str], harmful_margin: float = 0.01) -> V08Model:
    X = train_df[features].to_numpy(np.float64)
    y = train_df["utility_target"].to_numpy(np.float64)
    u = train_df["forecast_utility"].to_numpy(np.float64)
    yb = (u > 0.0).astype(np.int64)
    yh = (u < -float(harmful_margin)).astype(np.int64)
    if spec.task == "classification":
        spec.estimator = _fit_binary(spec.estimator, X, yb)
        return spec
    if spec.task == "cost_sensitive":
        if np.unique(yb).size < 2:
            spec.estimator = float(np.mean(yb)); return spec
        # Harmful actions have much larger downside in v0.6/v0.7. Weight them by both class and magnitude.
        scale = max(float(np.nanmedian(np.abs(y[np.isfinite(y)]))), 1e-4)
        w = np.ones(len(y), np.float64)
        w[yh == 1] += 2.0 + np.minimum(np.abs(y[yh == 1]) / scale, 5.0)
        w[yb == 1] += 0.5
        spec.estimator.fit(X, yb, model__sample_weight=w)
        return spec
    if spec.task == "regression":
        spec.estimator.fit(X, y)
        return spec
    if spec.task == "risk_expected":
        spec.benefit_estimator = _fit_binary(spec.benefit_estimator, X, yb)
        spec.harm_estimator = _fit_binary(spec.harm_estimator, X, yh)
        spec.positive_estimator = _fit_subset_reg(spec.positive_estimator, X, np.maximum(y, 0.0), yb == 1)
        spec.harmful_magnitude_estimator = _fit_subset_reg(
            spec.harmful_magnitude_estimator, X, np.maximum(-y, 0.0), yh == 1
        )
        return spec
    raise KeyError(spec.task)


def _prob(est, X):
    if isinstance(est, float):
        return np.full(len(X), float(est), np.float64)
    return np.asarray(est.predict_proba(X)[:, 1], np.float64)


def _reg(est, X):
    if isinstance(est, float):
        return np.full(len(X), float(est), np.float64)
    return np.asarray(est.predict(X), np.float64)


def predict_v08_scores(spec: V08Model, df: pd.DataFrame, features: List[str]) -> np.ndarray:
    X = df[features].to_numpy(np.float64)
    if spec.task in {"classification", "cost_sensitive"}:
        return _prob(spec.estimator, X)
    if spec.task == "regression":
        return _reg(spec.estimator, X)
    if spec.task == "risk_expected":
        pb = _prob(spec.benefit_estimator, X)
        ph = _prob(spec.harm_estimator, X)
        pos = np.maximum(_reg(spec.positive_estimator, X), 0.0)
        harm = np.maximum(_reg(spec.harmful_magnitude_estimator, X), 0.0)
        return pb * pos - ph * harm
    raise KeyError(spec.task)


def action_metrics_v08(df: pd.DataFrame, scores: np.ndarray) -> Dict[str, float]:
    u = df["forecast_utility"].to_numpy(np.float64)
    y = (u > 0.0).astype(np.int64)
    s = np.asarray(scores, np.float64)
    good = np.isfinite(s)
    out = {
        "action_rows": int(good.sum()),
        "beneficial_rate": float(np.mean(y[good])) if np.any(good) else float("nan"),
        "mean_true_utility": float(np.mean(u[good])) if np.any(good) else float("nan"),
    }
    if np.any(good) and np.unique(y[good]).size > 1:
        out["action_roc_auc"] = float(roc_auc_score(y[good], s[good]))
        out["action_average_precision"] = float(average_precision_score(y[good], s[good]))
    else:
        out["action_roc_auc"] = float("nan")
        out["action_average_precision"] = float("nan")
    return out


def parse_csv_list(text: str | None, default: Iterable[str]) -> List[str]:
    if text is None:
        return list(default)
    return [x.strip() for x in str(text).split(",") if x.strip()]
