from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import re
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.base import clone
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover - surfaced by preflight
    raise RuntimeError(
        "scikit-learn is required for SRD v0.4 realizable benchmark. "
        "Install with: pip install scikit-learn>=1.3"
    ) from exc


DEFAULT_SCOPES = [1, 2, 3, 4, 6]
DEFAULT_MODELS = ["similarity_rule", "logistic_benefit", "ridge_utility", "histgb_utility", "mlp_utility"]


def parse_int_list(text: str | None, default: Iterable[int]) -> List[int]:
    if text is None:
        return list(default)
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_str_list(text: str | None, default: Iterable[str]) -> List[str]:
    if text is None:
        return list(default)
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _v(r, key, i, j, default=np.nan):
    return float(r[key][i, j]) if key in r else float(default)


def _scope_steps(value) -> int:
    m = re.search(r"recent\s*(\d+)", str(value).lower())
    if not m:
        raise ValueError(f"Cannot parse restoration scope: {value}")
    return int(m.group(1))


def _build_scope_table_from_npz(source_run: Path, scope_steps: int, max_k: int) -> pd.DataFrame:
    retrieval_path = source_run / "retrieval" / "retrieval_topk.npz"
    eval_path = source_run / f"oracle_raw_copy_recent{scope_steps}" / "candidate_eval.npz"
    if not retrieval_path.exists() or not eval_path.exists():
        raise FileNotFoundError(
            f"Missing v0.3 artifacts for recent{scope_steps}: {retrieval_path} / {eval_path}"
        )
    r = np.load(retrieval_path)
    e = np.load(eval_path)
    q = r["query_ids"].astype(np.int64)
    if not np.array_equal(q, e["query_ids"].astype(np.int64)):
        raise RuntimeError(f"retrieval/eval query IDs differ for recent{scope_steps}")
    c = e["candidate_ids"].astype(np.int64)
    K = min(int(max_k), int(c.shape[1]))
    harm = 0.01
    rows = []
    for i, qi in enumerate(q):
        for j in range(K):
            base = float(e["baseline_mae"][i])
            model = float(e["candidate_mae"][i, j])
            utility = base - model
            rows.append({
                "query_id": int(qi),
                "candidate_rank": j + 1,
                "candidate_id": int(c[i, j]),
                "transport": "raw_copy",
                "restoration_scope": f"recent{scope_steps}",
                "retrieval_score": _v(r, "score", i, j),
                "latent_similarity": _v(r, "latent_sim", i, j),
                "observed_mae": _v(r, "observed_mae", i, j),
                "observed_mae_recent3": _v(r, "observed_mae_recent3", i, j),
                "observed_mae_recent6": _v(r, "observed_mae_recent6", i, j),
                "observed_mae_all": _v(r, "observed_mae_all", i, j),
                "context_score": _v(r, "context_score", i, j),
                "overlap_fraction": _v(r, "overlap_fraction", i, j),
                "overlap_recent3": _v(r, "overlap_recent3", i, j),
                "overlap_recent6": _v(r, "overlap_recent6", i, j),
                "baseline_masked_mae": base,
                "restored_masked_mae": model,
                "utility": utility,
                "beneficial": int(utility > 0),
                "harmful": int(utility < -harm),
                "fill_count": int(e["fill_counts"][i, j]),
            })
    return pd.DataFrame(rows)


def load_action_table(source_run: str | Path, scopes: Iterable[int], max_k: int) -> pd.DataFrame:
    source_run = Path(source_run).resolve()
    frames = []
    for L in scopes:
        csv_path = source_run / f"oracle_raw_copy_recent{int(L)}" / "utility_dataset.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df = df[df["candidate_rank"] <= int(max_k)].copy()
        else:
            df = _build_scope_table_from_npz(source_run, int(L), int(max_k))
        if df.empty:
            raise RuntimeError(f"No utility rows for recent{L}")
        df["scope_steps"] = int(L)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out[out["candidate_rank"] <= int(max_k)].copy()
    out["query_id"] = out["query_id"].astype(np.int64)
    out["candidate_rank"] = out["candidate_rank"].astype(np.int64)
    out["scope_steps"] = out["scope_steps"].astype(np.int64)
    # Basic integrity checks.
    dup = out.duplicated(["query_id", "candidate_rank", "scope_steps"]).sum()
    if dup:
        raise RuntimeError(f"Duplicate query/rank/scope actions: {dup}")
    base_nunique = out.groupby("query_id")["baseline_masked_mae"].nunique(dropna=False)
    if int((base_nunique > 1).sum()) > 0:
        raise RuntimeError("Baseline MAE differs across actions for the same query")
    return add_features(out)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for col in [
        "retrieval_score", "latent_similarity", "observed_mae", "observed_mae_recent3",
        "observed_mae_recent6", "observed_mae_all", "context_score", "overlap_fraction",
        "overlap_recent3", "overlap_recent6", "fill_count",
    ]:
        if col not in x:
            x[col] = np.nan
    x["latent_distance"] = 1.0 - x["latent_similarity"]
    x["recent3_minus_all"] = x["observed_mae_recent3"] - x["observed_mae_all"]
    x["recent6_minus_all"] = x["observed_mae_recent6"] - x["observed_mae_all"]
    x["recent3_minus_recent6"] = x["observed_mae_recent3"] - x["observed_mae_recent6"]
    x["candidate_rank_inv"] = 1.0 / np.maximum(x["candidate_rank"].astype(float), 1.0)
    x["candidate_rank_log"] = np.log1p(x["candidate_rank"].astype(float))
    x["fill_count_log"] = np.log1p(np.maximum(x["fill_count"].astype(float), 0.0))
    x["scope_steps_float"] = x["scope_steps"].astype(float)
    for L in sorted(x["scope_steps"].unique().tolist()):
        x[f"scope_is_{int(L)}"] = (x["scope_steps"] == int(L)).astype(float)
    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    return x


def feature_columns(df: pd.DataFrame) -> List[str]:
    cols = [
        "retrieval_score",
        "latent_similarity",
        "latent_distance",
        "observed_mae_recent3",
        "observed_mae_recent6",
        "observed_mae_all",
        "recent3_minus_all",
        "recent6_minus_all",
        "recent3_minus_recent6",
        "context_score",
        "overlap_fraction",
        "overlap_recent3",
        "overlap_recent6",
        "candidate_rank_inv",
        "candidate_rank_log",
        "fill_count_log",
        "scope_steps_float",
    ]
    cols += sorted([c for c in df.columns if c.startswith("scope_is_")])
    # Explicitly exclude anything requiring future targets such as baseline/restored MAE and utility.
    return [c for c in cols if c in df.columns]


@dataclass
class ModelSpec:
    name: str
    task: str  # rule | classification | regression
    estimator: object | None = None


def make_model(name: str, seed: int) -> ModelSpec:
    name = str(name)
    if name == "similarity_rule":
        return ModelSpec(name=name, task="rule", estimator=None)
    if name == "logistic_benefit":
        est = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=800, class_weight="balanced", random_state=seed)),
        ])
        return ModelSpec(name=name, task="classification", estimator=est)
    if name == "ridge_utility":
        est = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])
        return ModelSpec(name=name, task="regression", estimator=est)
    if name == "histgb_utility":
        est = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                max_iter=180,
                max_depth=4,
                learning_rate=0.05,
                l2_regularization=1.0,
                random_state=seed,
            )),
        ])
        return ModelSpec(name=name, task="regression", estimator=est)
    if name == "mlp_utility":
        est = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=160,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=12,
                random_state=seed,
            )),
        ])
        return ModelSpec(name=name, task="regression", estimator=est)
    raise ValueError(f"Unknown realizable model: {name}")


def fit_model(spec: ModelSpec, train_df: pd.DataFrame, features: List[str]):
    if spec.task == "rule":
        return spec
    X = train_df[features].to_numpy(dtype=np.float64)
    if spec.task == "classification":
        y = train_df["beneficial"].to_numpy(dtype=np.int64)
        if np.unique(y).size < 2:
            # Rare defensive fallback. A constant-probability rule keeps the benchmark running.
            return ModelSpec(name=spec.name, task="constant", estimator=float(np.mean(y)))
    else:
        y = train_df["utility"].to_numpy(dtype=np.float64)
    est = clone(spec.estimator)
    est.fit(X, y)
    return ModelSpec(name=spec.name, task=spec.task, estimator=est)


def predict_scores(spec: ModelSpec, df: pd.DataFrame, features: List[str]) -> np.ndarray:
    if spec.task == "rule":
        # Same retrieval score is repeated across restoration scopes. Break ties toward
        # the longer recent suffix (v0.3's strongest fixed-scope raw-copy baseline)
        # without using any future target label.
        return (
            df["retrieval_score"].to_numpy(dtype=np.float64)
            + 1e-9 * df["scope_steps"].to_numpy(dtype=np.float64)
        )
    if spec.task == "constant":
        return np.full(len(df), float(spec.estimator), dtype=np.float64)
    X = df[features].to_numpy(dtype=np.float64)
    if spec.task == "classification":
        return np.asarray(spec.estimator.predict_proba(X)[:, 1], dtype=np.float64)
    return np.asarray(spec.estimator.predict(X), dtype=np.float64)


def query_folds(query_ids: Iterable[int], n_splits: int, protocol: str, seed: int):
    qids = np.asarray(sorted(set(map(int, query_ids))), dtype=np.int64)
    n_splits = min(max(2, int(n_splits)), len(qids))
    if protocol == "blocked":
        blocks = [np.asarray(x, dtype=np.int64) for x in np.array_split(qids, n_splits) if len(x)]
        for fold, test in enumerate(blocks):
            train = np.setdiff1d(qids, test, assume_unique=True)
            yield fold, train, test
        return
    if protocol == "random":
        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(qids)
        blocks = [np.asarray(x, dtype=np.int64) for x in np.array_split(perm, n_splits) if len(x)]
        for fold, test in enumerate(blocks):
            train = np.setdiff1d(qids, test, assume_unique=True)
            yield fold, train, test
        return
    raise ValueError(f"Unknown CV protocol: {protocol}")


def fit_calib_queries(train_qids: np.ndarray, calib_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    train_qids = np.asarray(train_qids, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(train_qids)
    n_cal = max(1, int(round(len(perm) * float(calib_fraction))))
    n_cal = min(n_cal, max(1, len(perm) - 2))
    calib = np.sort(perm[:n_cal])
    fit = np.sort(perm[n_cal:])
    if len(fit) < 2:
        fit = np.sort(perm[:-1])
        calib = np.sort(perm[-1:])
    return fit, calib


def _best_action_rows(df: pd.DataFrame, score_col: str = "_score") -> pd.DataFrame:
    # Stable deterministic tie-break: higher score, then smaller candidate rank, then shorter scope.
    work = df.sort_values(
        ["query_id", score_col, "candidate_rank", "scope_steps"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    return work.groupby("query_id", as_index=False, sort=True).head(1).copy()


def tune_threshold(calib_df: pd.DataFrame, scores: np.ndarray) -> Tuple[float, Dict[str, float]]:
    c = calib_df.copy()
    c["_score"] = np.asarray(scores, dtype=np.float64)
    top = _best_action_rows(c)
    vals = top["_score"].to_numpy(dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return float("inf"), {"mean_utility": 0.0, "intervention_rate": 0.0}
    qs = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 101)))
    thresholds = np.concatenate(([-np.inf], qs, [np.inf]))
    best = None
    for tau in thresholds:
        use = top["_score"].to_numpy(dtype=np.float64) > float(tau)
        util = np.where(use, top["utility"].to_numpy(dtype=np.float64), 0.0)
        mean_u = float(np.mean(util))
        ir = float(np.mean(use))
        # Maximize realized utility; ties prefer a more conservative (higher) threshold.
        key = (mean_u, float(tau))
        if best is None or key > best[0]:
            best = (key, float(tau), mean_u, ir)
    assert best is not None
    return best[1], {"mean_utility": best[2], "intervention_rate": best[3]}


def _query_baseline_map(df: pd.DataFrame) -> pd.Series:
    return df.groupby("query_id", sort=True)["baseline_masked_mae"].first()


def oracle_metrics(df: pd.DataFrame, harmful_margin: float) -> Dict[str, float]:
    base = _query_baseline_map(df)
    best_u = df.groupby("query_id", sort=True)["utility"].max().reindex(base.index)
    safe_u = np.maximum(best_u.to_numpy(dtype=np.float64), 0.0)
    b = base.to_numpy(dtype=np.float64)
    oracle_mae = b - safe_u
    bmean = float(np.mean(b))
    mean = float(np.mean(oracle_mae))
    gain = (bmean - mean) / bmean * 100.0 if bmean else float("nan")
    return {
        "queries": int(len(base)),
        "baseline_mae": bmean,
        "oracle_mae": mean,
        "oracle_gain_pct": gain,
        "candidate_hit_rate": float(np.mean(best_u.to_numpy(dtype=np.float64) > 0.0)),
        "no_intervention_rate": float(np.mean(best_u.to_numpy(dtype=np.float64) <= 0.0)),
    }


def evaluate_decisions(test_df: pd.DataFrame, scores: np.ndarray, threshold: float, harmful_margin: float):
    d = test_df.copy()
    d["_score"] = np.asarray(scores, dtype=np.float64)
    top = _best_action_rows(d)
    base = _query_baseline_map(d)
    top = top.set_index("query_id").reindex(base.index)
    score = top["_score"].to_numpy(dtype=np.float64)
    actual_u = top["utility"].to_numpy(dtype=np.float64)
    use = np.isfinite(score) & (score > float(threshold))
    realized_u = np.where(use, actual_u, 0.0)
    b = base.to_numpy(dtype=np.float64)
    realized_mae = b - realized_u
    bmean = float(np.mean(b))
    rmean = float(np.mean(realized_mae))
    gain = (bmean - rmean) / bmean * 100.0 if bmean else float("nan")

    # Ranking-only diagnostic: learned ranking, but an oracle gate blocks non-beneficial actions.
    ranking_only_u = np.maximum(actual_u, 0.0)
    ranking_only_mae = b - ranking_only_u
    ranking_only_gain = (bmean - float(np.mean(ranking_only_mae))) / bmean * 100.0 if bmean else float("nan")

    # Gate-only diagnostic: oracle gives the best action, learned score/threshold decides whether to use it.
    idx_oracle = d.sort_values(["query_id", "utility"], ascending=[True, False], kind="mergesort").groupby("query_id", as_index=False).head(1)
    idx_oracle = idx_oracle.set_index("query_id").reindex(base.index)
    oracle_u_raw = idx_oracle["utility"].to_numpy(dtype=np.float64)
    oracle_score = idx_oracle["_score"].to_numpy(dtype=np.float64)
    gate_use = np.isfinite(oracle_score) & (oracle_score > float(threshold))
    gate_only_u = np.where(gate_use, oracle_u_raw, 0.0)
    gate_only_mae = b - gate_only_u
    gate_only_gain = (bmean - float(np.mean(gate_only_mae))) / bmean * 100.0 if bmean else float("nan")

    best_u = d.groupby("query_id", sort=True)["utility"].max().reindex(base.index).to_numpy(dtype=np.float64)
    oracle_u = np.maximum(best_u, 0.0)
    oracle_mae = b - oracle_u
    oracle_mean = float(np.mean(oracle_mae))
    oracle_gain = (bmean - oracle_mean) / bmean * 100.0 if bmean else float("nan")
    conv = gain / oracle_gain if oracle_gain > 1e-12 else float("nan")

    n_use = int(use.sum())
    beneficial_precision = float(np.mean(actual_u[use] > 0.0)) if n_use else float("nan")
    harmful_rate = float(np.mean(actual_u[use] < -float(harmful_margin))) if n_use else 0.0
    neutral_or_better = float(np.mean(actual_u[use] >= -float(harmful_margin))) if n_use else 1.0

    decisions = pd.DataFrame({
        "query_id": base.index.to_numpy(dtype=np.int64),
        "baseline_mae": b,
        "selected_candidate_rank": top["candidate_rank"].to_numpy(dtype=np.int64),
        "selected_candidate_id": top["candidate_id"].to_numpy(dtype=np.int64),
        "selected_scope_steps": top["scope_steps"].to_numpy(dtype=np.int64),
        "selected_score": score,
        "threshold": float(threshold),
        "intervene": use.astype(np.int64),
        "selected_actual_utility": actual_u,
        "realized_utility": realized_u,
        "realized_mae": realized_mae,
        "oracle_utility": oracle_u,
        "ranking_only_utility": ranking_only_u,
        "ranking_only_mae": ranking_only_mae,
        "gate_only_utility": gate_only_u,
        "gate_only_mae": gate_only_mae,
    })
    metrics = {
        "queries": int(len(base)),
        "baseline_mae": bmean,
        "realized_mae": rmean,
        "realized_gain_pct": gain,
        "oracle_mae": oracle_mean,
        "oracle_gain_pct": oracle_gain,
        "oracle_conversion_ratio": conv,
        "intervention_rate": float(np.mean(use)),
        "beneficial_precision": beneficial_precision,
        "harmful_intervention_rate": harmful_rate,
        "nonharmful_intervention_rate": neutral_or_better,
        "mean_realized_utility": float(np.mean(realized_u)),
        "learned_ranking_oracle_gate_gain_pct": ranking_only_gain,
        "oracle_ranking_learned_gate_gain_pct": gate_only_gain,
        "selected_action_beneficial_rate_all_queries": float(np.mean(actual_u > 0.0)),
    }
    return metrics, decisions


def action_level_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    good = np.isfinite(s)
    y = y[good]; s = s[good]
    out = {"action_rows": int(len(y))}
    if len(y) and np.unique(y).size >= 2:
        out["action_roc_auc"] = float(roc_auc_score(y, s))
        out["action_average_precision"] = float(average_precision_score(y, s))
    else:
        out["action_roc_auc"] = float("nan")
        out["action_average_precision"] = float("nan")
    return out


def source_cross_scope_gain(source_run: Path, max_k: int) -> float | None:
    p = source_run / "window_ceiling_sweep" / "cross_scope_oracle.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    if d.empty:
        return None
    rows = d[d["k"] <= int(max_k)]
    if rows.empty:
        return None
    row = rows.sort_values("k").iloc[-1]
    return float(row["cross_scope_oracle_gain_pct"])


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(obj), indent=2, sort_keys=True, default=str, allow_nan=False), encoding="utf-8")
