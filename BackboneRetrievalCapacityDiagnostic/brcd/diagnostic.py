from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .backbone_bridge import (
    MODELS,
    assert_prediction_alignment,
    select_prediction_rows,
    cache_path,
    discover_formal_artifacts,
    discover_test_predictions,
    load_prediction_npz,
)
from .data import RetrievalSplit, load_split
from .metrics import (
    fuse,
    gain_pct,
    oracle_query_fusion_mae,
    paired_bootstrap_gain,
    per_query_mae,
    spearman,
    summary,
)
from .retrieval import TopKResult, aggregate_future, build_embedding, oracle_future, search_topk
from .utils import ensure_dir, log, project_root, write_json


def _save_topk(path: Path, topk: TopKResult, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, indices=topk.indices, distances=topk.distances, pool_sizes=topk.pool_sizes)
    write_json(path.with_suffix(".json"), metadata)


def _save_prediction(path: Path, pred: np.ndarray, ids: np.ndarray, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, view_id=np.asarray(ids).astype("U"), prediction=np.asarray(pred, dtype=np.float32))
    write_json(path.with_suffix(".json"), metadata)


def _load_backbone_val_predictions(target: str, dataset: str, seed: int, models: list[str], val: RetrievalSplit):
    out = {}
    arts = {}
    for model in models:
        art = discover_formal_artifacts(target, model, dataset, seed)
        pred, ids = load_prediction_npz(art.val_predictions)
        pred, ids = select_prediction_rows(
            pred, ids, val.view_ids, f"{target}/{model}/validation",
            allow_prefix_without_ids=True,
        )
        assert_prediction_alignment(pred, ids, val.view_ids, f"{target}/{model}/validation")
        out[model] = pred
        arts[model] = art
        log(f"BACKBONE_VAL model={model} formal={art.formal_dir.name} pred={art.val_predictions}")
    return out, arts


def _retrieval_grid(train: RetrievalSplit, val: RetrievalSplit, topk: TopKResult,
                    k_grid: list[int], tau_grid: list[float], chunk_size: int):
    rows = []
    best = None
    best_pred = None
    for tau in tau_grid:
        for k in k_grid:
            pred, _ = aggregate_future(topk, train.y, train.future_valid, k, tau, chunk_size=chunk_size)
            s = summary(pred, val.y, val.future_valid, val.direct)
            row = {"k": int(k), "tau": float(tau), **s}
            rows.append(row)
            log(f"RETRIEVAL_CAL k={k} tau={tau:g} MAE={s['MAE']:.6f} Masked_MAE={s['Masked_MAE']:.6f}")
            key = (float(s["MAE"]), int(k), float(tau))
            if best is None or key < best:
                best = key
                best_pred = pred.copy()
                best_row = row.copy()
    return pd.DataFrame(rows), best_row, best_pred


def _calibrate_alphas(backbone_preds: dict[str, np.ndarray], retrieval_pred: np.ndarray, val: RetrievalSplit,
                      alpha_grid: list[float]):
    rows = []
    best_by_model = {}
    for model, base in backbone_preds.items():
        bsum = summary(base, val.y, val.future_valid, val.direct)
        model_rows = []
        for alpha in alpha_grid:
            pred = fuse(base, retrieval_pred, alpha)
            s = summary(pred, val.y, val.future_valid, val.direct)
            row = {
                "model": model, "alpha": float(alpha),
                "base_MAE": bsum["MAE"], "base_Masked_MAE": bsum["Masked_MAE"],
                **s,
                "gain_pct": gain_pct(bsum["MAE"], s["MAE"]),
                "masked_gain_pct": gain_pct(bsum["Masked_MAE"], s["Masked_MAE"]),
            }
            model_rows.append(row); rows.append(row)
        best = min(model_rows, key=lambda r: (r["MAE"], r["alpha"]))
        best_by_model[model] = best
        log(f"ALPHA_CAL model={model} alpha={best['alpha']:.2f} val_gain={best['gain_pct']:.4f}%")

    # Strict control: one shared alpha for all backbones, selected by mean relative validation MAE.
    shared_rows = []
    for alpha in alpha_grid:
        sub = [r for r in rows if abs(r["alpha"] - alpha) < 1e-12]
        score = float(np.mean([r["MAE"] / max(r["base_MAE"], 1e-12) for r in sub]))
        masked_score = float(np.mean([r["Masked_MAE"] / max(r["base_Masked_MAE"], 1e-12) for r in sub]))
        shared_rows.append({"alpha": float(alpha), "mean_relative_MAE": score, "mean_relative_Masked_MAE": masked_score})
    shared_best = min(shared_rows, key=lambda r: (r["mean_relative_MAE"], r["alpha"]))
    log(f"SHARED_ALPHA alpha={shared_best['alpha']:.2f} mean_relative_MAE={shared_best['mean_relative_MAE']:.6f}")
    return pd.DataFrame(rows), pd.DataFrame(shared_rows), best_by_model, shared_best


def run(cfg: dict, *, target: str, severities: list[int], models: list[str], seed: int,
        run_dir: Path, max_queries: int | None = None, max_memory: int | None = None,
        bootstrap_samples: int | None = None) -> dict:
    exp = cfg["experiment"]
    ret = cfg["retrieval"]
    fus = cfg["fusion"]
    ana = cfg["analysis"]
    dataset = str(exp["dataset"])
    models = [m for m in models if m in MODELS]
    if not models:
        raise ValueError("No supported backbones selected")
    ensure_dir(run_dir)

    formal_cache = cache_path(target, "formal")
    train = load_split(formal_cache, "train", max_queries=max_memory)
    val = load_split(formal_cache, "val", max_queries=max_queries)
    log(f"DATA target={target} train={train.q} val={val.q} H={train.h} P={train.p} N={train.n} mask={train.mask_path.name}")

    train_emb = build_embedding(train.x, train.observed, ret["temporal_bins"], ret["mask_weight"], ret["embedding_chunk_size"])
    val_emb = build_embedding(val.x, val.observed, ret["temporal_bins"], ret["mask_weight"], ret["embedding_chunk_size"])
    val_topk = search_topk(
        train_emb, val_emb, train.tod_hist, train.dow_hist, val.tod_hist, val.dow_hist,
        kmax=max(ret["k_grid"]), tod_radius=ret["tod_radius"], tod_slots=ret["tod_slots"],
        match_day_type=ret["match_day_type"], batch_queries=ret["search_batch_queries"],
    )
    _save_topk(run_dir / "calibration" / "val_topk.npz", val_topk, {
        "target": target, "split": "val", "memory_split": "train", "memory_rows": train.q,
        "inference_safe": True, "uses_backbone_in_retrieval": False,
    })

    grid_df, best_ret, val_retrieval = _retrieval_grid(
        train, val, val_topk, [int(x) for x in ret["k_grid"]], [float(x) for x in ret["tau_grid"]], ret["aggregation_chunk_size"]
    )
    grid_df.to_csv(run_dir / "calibration" / "retrieval_grid.csv", index=False)
    write_json(run_dir / "calibration" / "retrieval_best.json", best_ret)
    _save_prediction(run_dir / "calibration" / "val_retrieval_predictions.npz", val_retrieval, val.view_ids, best_ret)

    backbone_val, arts = _load_backbone_val_predictions(target, dataset, seed, models, val)
    alpha_grid = [float(x) for x in fus["alpha_grid"]]
    alpha_df, shared_df, best_alpha, shared_best = _calibrate_alphas(backbone_val, val_retrieval, val, alpha_grid)
    alpha_df.to_csv(run_dir / "calibration" / "alpha_grid.csv", index=False)
    shared_df.to_csv(run_dir / "calibration" / "shared_alpha_grid.csv", index=False)
    pd.DataFrame(best_alpha.values()).to_csv(run_dir / "calibration" / "backbone_alpha.csv", index=False)
    write_json(run_dir / "calibration" / "shared_alpha.json", shared_best)

    all_rows = []
    correlation_rows = []
    oracle_k = int(ana.get("oracle_k", max(ret["k_grid"])))
    n_boot = int(bootstrap_samples if bootstrap_samples is not None else ana["bootstrap_samples"])

    for severity in severities:
        test_cache = cache_path(target, "evaluation", int(severity))
        test = load_split(test_cache, "test", max_queries=max_queries)
        test_emb = build_embedding(test.x, test.observed, ret["temporal_bins"], ret["mask_weight"], ret["embedding_chunk_size"])
        topk = search_topk(
            train_emb, test_emb, train.tod_hist, train.dow_hist, test.tod_hist, test.dow_hist,
            kmax=max(ret["k_grid"]), tod_radius=ret["tod_radius"], tod_slots=ret["tod_slots"],
            match_day_type=ret["match_day_type"], batch_queries=ret["search_batch_queries"],
        )
        sev_dir = ensure_dir(run_dir / f"s{int(severity)}")
        _save_topk(sev_dir / "topk.npz", topk, {"severity": int(severity), "target": target})
        retrieval_pred, retrieval_valid = aggregate_future(
            topk, train.y, train.future_valid, int(best_ret["k"]), float(best_ret["tau"]),
            chunk_size=ret["aggregation_chunk_size"],
        )
        _save_prediction(sev_dir / "retrieval_predictions.npz", retrieval_pred, test.view_ids, best_ret)
        ret_sum = summary(retrieval_pred, test.y, test.future_valid, test.direct)

        oracle_all, _, chosen_all = oracle_future(
            topk, train.y, train.future_valid, test.y, test.future_valid, oracle_k,
            extra_mask=None, chunk_size=ana["oracle_chunk_size"],
        )
        oracle_masked, _, chosen_masked = oracle_future(
            topk, train.y, train.future_valid, test.y, test.future_valid, oracle_k,
            extra_mask=test.direct, chunk_size=ana["oracle_chunk_size"],
        )
        oracle_all_sum = summary(oracle_all, test.y, test.future_valid, test.direct)
        oracle_masked_sum = summary(oracle_masked, test.y, test.future_valid, test.direct)
        np.savez_compressed(sev_dir / "oracle_choices.npz", all_candidate_index=chosen_all, masked_candidate_index=chosen_masked)

        for model in models:
            pred_path = discover_test_predictions(arts[model], dataset, int(severity))
            base, ids = load_prediction_npz(pred_path)
            base, ids = select_prediction_rows(
                base, ids, test.view_ids, f"{target}/{model}/s{severity}",
                allow_prefix_without_ids=True,
            )
            assert_prediction_alignment(base, ids, test.view_ids, f"{target}/{model}/s{severity}")
            bsum = summary(base, test.y, test.future_valid, test.direct)

            a_model = float(best_alpha[model]["alpha"])
            a_shared = float(shared_best["alpha"])
            p_model = fuse(base, retrieval_pred, a_model)
            p_shared = fuse(base, retrieval_pred, a_shared)
            msum = summary(p_model, test.y, test.future_valid, test.direct)
            ssum = summary(p_shared, test.y, test.future_valid, test.direct)

            bq = per_query_mae(base, test.y, test.future_valid)
            mq = per_query_mae(p_model, test.y, test.future_valid)
            bqm = per_query_mae(base, test.y, test.future_valid, extra=test.direct)
            mqm = per_query_mae(p_model, test.y, test.future_valid, extra=test.direct)
            ci = paired_bootstrap_gain(bq, mq, n_boot=n_boot, seed=seed + int(severity))
            cim = paired_bootstrap_gain(bqm, mqm, n_boot=n_boot, seed=seed + int(severity) + 1000)

            oracle_fusion_mae = oracle_query_fusion_mae(
                base, oracle_all, test.y, test.future_valid, alpha_grid, extra=None
            )
            oracle_fusion_masked_mae = oracle_query_fusion_mae(
                base, oracle_masked, test.y, test.future_valid, alpha_grid, extra=test.direct
            )
            row = {
                "target": target, "severity": int(severity), "model": model,
                "formal_run": arts[model].formal_dir.name,
                "retrieval_k": int(best_ret["k"]), "retrieval_tau": float(best_ret["tau"]),
                "retrieval_MAE": ret_sum["MAE"], "retrieval_Masked_MAE": ret_sum["Masked_MAE"],
                "base_MAE": bsum["MAE"], "base_RMSE": bsum["RMSE"],
                "base_Masked_MAE": bsum["Masked_MAE"], "base_Masked_RMSE": bsum["Masked_RMSE"],
                "alpha_model": a_model,
                "fused_MAE": msum["MAE"], "fused_RMSE": msum["RMSE"],
                "fused_Masked_MAE": msum["Masked_MAE"], "fused_Masked_RMSE": msum["Masked_RMSE"],
                "gain_pct": gain_pct(bsum["MAE"], msum["MAE"]),
                "masked_gain_pct": gain_pct(bsum["Masked_MAE"], msum["Masked_MAE"]),
                "gain_ci_low": ci["ci_low"], "gain_ci_high": ci["ci_high"],
                "masked_gain_ci_low": cim["ci_low"], "masked_gain_ci_high": cim["ci_high"],
                "alpha_shared": a_shared,
                "shared_fused_MAE": ssum["MAE"], "shared_gain_pct": gain_pct(bsum["MAE"], ssum["MAE"]),
                "shared_fused_Masked_MAE": ssum["Masked_MAE"],
                "shared_masked_gain_pct": gain_pct(bsum["Masked_MAE"], ssum["Masked_MAE"]),
                "oracle_retrieval_MAE_at_k": oracle_all_sum["MAE"],
                "oracle_retrieval_Masked_MAE_at_k": oracle_masked_sum["Masked_MAE"],
                "oracle_retrieval_gain_pct": gain_pct(bsum["MAE"], oracle_all_sum["MAE"]),
                "oracle_retrieval_masked_gain_pct": gain_pct(bsum["Masked_MAE"], oracle_masked_sum["Masked_MAE"]),
                "oracle_query_fusion_MAE": oracle_fusion_mae,
                "oracle_query_fusion_gain_pct": gain_pct(bsum["MAE"], oracle_fusion_mae),
                "oracle_query_fusion_Masked_MAE": oracle_fusion_masked_mae,
                "oracle_query_fusion_masked_gain_pct": gain_pct(bsum["Masked_MAE"], oracle_fusion_masked_mae),
                "test_prediction_path": str(pred_path),
            }
            all_rows.append(row)
            log(
                f"RESULT target={target} s{severity} model={model} base={bsum['MAE']:.6f} "
                f"fused={msum['MAE']:.6f} gain={row['gain_pct']:+.3f}% alpha={a_model:.2f} "
                f"oracleFusionGain={row['oracle_query_fusion_gain_pct']:+.3f}%"
            )

        sev_rows = [r for r in all_rows if r["severity"] == int(severity)]
        correlation_rows.append({
            "target": target, "severity": int(severity),
            "spearman_base_MAE_vs_gain": spearman([r["base_MAE"] for r in sev_rows], [r["gain_pct"] for r in sev_rows]),
            "spearman_base_MAE_vs_shared_gain": spearman([r["base_MAE"] for r in sev_rows], [r["shared_gain_pct"] for r in sev_rows]),
            "spearman_base_Masked_MAE_vs_masked_gain": spearman([r["base_Masked_MAE"] for r in sev_rows], [r["masked_gain_pct"] for r in sev_rows]),
            "n_models": len(sev_rows),
        })
        pd.DataFrame(sev_rows).to_csv(sev_dir / "capacity_results.csv", index=False)

    results = pd.DataFrame(all_rows)
    corr = pd.DataFrame(correlation_rows)
    results.to_csv(run_dir / "capacity_results.csv", index=False)
    corr.to_csv(run_dir / "capacity_correlations.csv", index=False)
    report = {
        "target": target, "dataset": dataset, "seed": int(seed), "models": models,
        "severities": [int(x) for x in severities], "formal_cache": str(formal_cache),
        "train_memory_rows": int(train.q), "validation_rows": int(val.q),
        "retrieval_best": best_ret, "shared_alpha": shared_best,
        "backbone_alpha": {m: float(best_alpha[m]["alpha"]) for m in models},
        "uses_backbone_in_retrieval": False,
        "retrieval_inputs": ["x_obs_norm", "history_observed", "tod_hist", "dow_hist"],
        "forbidden_retrieval_inputs": ["future target", "direct_mask", "backbone prediction", "backbone latent"],
        "correlations": correlation_rows,
    }
    write_json(run_dir / "REPORT.json", report)
    return report
