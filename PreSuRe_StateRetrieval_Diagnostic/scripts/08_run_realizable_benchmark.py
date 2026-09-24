#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from srd.config import load_config
from srd.realizable import (
    DEFAULT_MODELS,
    DEFAULT_SCOPES,
    action_level_metrics,
    evaluate_decisions,
    feature_columns,
    fit_calib_queries,
    fit_model,
    load_action_table,
    make_model,
    oracle_metrics,
    parse_int_list,
    parse_str_list,
    predict_scores,
    query_folds,
    save_json,
    source_cross_scope_gain,
    tune_threshold,
)


def main():
    ap = argparse.ArgumentParser(description="SRD v0.4 realizable utility benchmark")
    ap.add_argument("--source-run", required=True, help="Completed SRD v0.3 experiment directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-k", type=int, default=50)
    ap.add_argument("--scopes", default="1,2,3,4,6")
    ap.add_argument("--protocols", default="blocked,random")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--calib-fraction", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--harmful-margin", type=float, default=0.01)
    a = ap.parse_args()

    source = Path(a.source_run).resolve()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    scopes = parse_int_list(a.scopes, DEFAULT_SCOPES)
    protocols = parse_str_list(a.protocols, ["blocked", "random"])
    models = parse_str_list(a.models, DEFAULT_MODELS)

    print("=" * 92)
    print("PreSuRe State-Retrieval Diagnostic v0.4 — Realizable Utility Benchmark")
    print("source_run       :", source)
    print("out              :", out)
    print("max_k            :", a.max_k)
    print("scopes           :", scopes)
    print("protocols        :", protocols)
    print("models           :", models)
    print("folds            :", a.folds)
    print("calib_fraction   :", a.calib_fraction)
    print("=" * 92)

    df = load_action_table(source, scopes=scopes, max_k=a.max_k)
    feats = feature_columns(df)
    qids = np.asarray(sorted(df.query_id.unique()), dtype=np.int64)
    print(f"actions={len(df):,} queries={len(qids)} features={len(feats)}")
    print("features:", feats)

    # No target-derived features are allowed.
    forbidden = {"utility", "beneficial", "harmful", "baseline_masked_mae", "restored_masked_mae"}
    leak = forbidden.intersection(feats)
    if leak:
        raise RuntimeError(f"Target leakage in features: {sorted(leak)}")

    global_oracle = oracle_metrics(df, harmful_margin=a.harmful_margin)
    source_gain = source_cross_scope_gain(source, a.max_k)
    if source_gain is not None:
        delta = float(global_oracle["oracle_gain_pct"] - source_gain)
        print(f"oracle sanity: reconstructed={global_oracle['oracle_gain_pct']:.6f}% source={source_gain:.6f}% delta={delta:+.6f} pp")
        if abs(delta) > 0.05:
            warnings.warn("Reconstructed oracle differs from v0.3 cross-scope oracle by >0.05 percentage points")

    pd.DataFrame({"feature": feats}).to_csv(out / "feature_schema.csv", index=False)
    df[[
        "query_id", "candidate_rank", "candidate_id", "scope_steps", "utility", "beneficial", "harmful"
    ]].to_csv(out / "action_index_and_labels.csv", index=False)

    all_summaries = []
    benchmark_summary = {
        "version": "0.4.0",
        "source_run": str(source),
        "max_k": int(a.max_k),
        "scopes": scopes,
        "protocols": protocols,
        "models": models,
        "folds": int(a.folds),
        "calib_fraction": float(a.calib_fraction),
        "seed": int(a.seed),
        "features": feats,
        "global_oracle": global_oracle,
        "source_cross_scope_oracle_gain_pct": source_gain,
        "results": {},
    }

    for protocol in protocols:
        print(f"\n### PROTOCOL {protocol} ###")
        benchmark_summary["results"][protocol] = {}
        for model_name in models:
            print(f"\n--- MODEL {model_name} ---")
            decision_parts = []
            thresholds = []
            fold_rows = []
            action_y = []
            action_s = []
            for fold, train_q, test_q in query_folds(qids, a.folds, protocol, a.seed):
                fit_q, calib_q = fit_calib_queries(train_q, a.calib_fraction, a.seed + 1009 * fold)
                fit_df = df[df.query_id.isin(fit_q)].copy()
                calib_df = df[df.query_id.isin(calib_q)].copy()
                test_df = df[df.query_id.isin(test_q)].copy()
                if fit_df.empty or calib_df.empty or test_df.empty:
                    raise RuntimeError(f"Empty fold partition protocol={protocol} fold={fold}")

                spec = make_model(model_name, seed=a.seed + fold)
                fitted = fit_model(spec, fit_df, feats)
                calib_scores = predict_scores(fitted, calib_df, feats)
                tau, calib_info = tune_threshold(calib_df, calib_scores)
                test_scores = predict_scores(fitted, test_df, feats)
                metrics, decisions = evaluate_decisions(
                    test_df, test_scores, threshold=tau, harmful_margin=a.harmful_margin
                )
                decisions.insert(1, "fold", fold)
                decisions["protocol"] = protocol
                decisions["model"] = model_name
                decision_parts.append(decisions)
                thresholds.append(float(tau))
                action_y.append(test_df["beneficial"].to_numpy(dtype=np.int64))
                action_s.append(np.asarray(test_scores, dtype=np.float64))
                fold_row = {
                    "protocol": protocol,
                    "model": model_name,
                    "fold": fold,
                    "fit_queries": int(len(fit_q)),
                    "calib_queries": int(len(calib_q)),
                    "test_queries": int(len(test_q)),
                    "threshold": float(tau),
                    "calib_mean_utility": float(calib_info["mean_utility"]),
                    "calib_intervention_rate": float(calib_info["intervention_rate"]),
                    **metrics,
                }
                fold_rows.append(fold_row)
                print(
                    f"fold={fold} tau={tau:.6g} realized={metrics['realized_gain_pct']:+.3f}% "
                    f"oracle={metrics['oracle_gain_pct']:+.3f}% conv={metrics['oracle_conversion_ratio']:.3f} "
                    f"IR={metrics['intervention_rate']:.3f} HIR={metrics['harmful_intervention_rate']:.3f}"
                )

            decisions = pd.concat(decision_parts, ignore_index=True).sort_values("query_id")
            # Aggregate OOF decisions globally instead of averaging fold percentages.
            b = decisions["baseline_mae"].to_numpy(dtype=np.float64)
            r = decisions["realized_mae"].to_numpy(dtype=np.float64)
            ou = decisions["oracle_utility"].to_numpy(dtype=np.float64)
            bmean = float(np.mean(b)); rmean = float(np.mean(r))
            realized_gain = (bmean - rmean) / bmean * 100.0 if bmean else float("nan")
            oracle_mae = b - ou
            oracle_mean = float(np.mean(oracle_mae))
            oracle_gain = (bmean - oracle_mean) / bmean * 100.0 if bmean else float("nan")
            conversion = realized_gain / oracle_gain if oracle_gain > 1e-12 else float("nan")
            use = decisions["intervene"].to_numpy(dtype=bool)
            sel_u = decisions["selected_actual_utility"].to_numpy(dtype=np.float64)
            n_use = int(use.sum())
            agg = {
                "protocol": protocol,
                "model": model_name,
                "queries": int(len(decisions)),
                "baseline_mae": bmean,
                "realized_mae": rmean,
                "realized_gain_pct": realized_gain,
                "oracle_mae": oracle_mean,
                "oracle_gain_pct": oracle_gain,
                "oracle_conversion_ratio": conversion,
                "intervention_rate": float(np.mean(use)),
                "beneficial_precision": float(np.mean(sel_u[use] > 0.0)) if n_use else float("nan"),
                "harmful_intervention_rate": float(np.mean(sel_u[use] < -a.harmful_margin)) if n_use else 0.0,
                "mean_threshold": float(np.mean(thresholds)),
                "median_threshold": float(np.median(thresholds)),
            }
            # Exact OOF decomposition: isolate ranking vs abstention/gating.
            fold_df = pd.DataFrame(fold_rows)
            rank_mean = float(np.mean(decisions["ranking_only_mae"].to_numpy(dtype=np.float64)))
            gate_mean = float(np.mean(decisions["gate_only_mae"].to_numpy(dtype=np.float64)))
            agg["learned_ranking_oracle_gate_gain_pct"] = (bmean - rank_mean) / bmean * 100.0 if bmean else float("nan")
            agg["oracle_ranking_learned_gate_gain_pct"] = (bmean - gate_mean) / bmean * 100.0 if bmean else float("nan")
            am = action_level_metrics(np.concatenate(action_y), np.concatenate(action_s))
            agg.update(am)

            model_dir = out / protocol / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            decisions.to_csv(model_dir / "oof_query_decisions.csv", index=False)
            fold_df.to_csv(model_dir / "fold_metrics.csv", index=False)
            save_json(model_dir / "summary.json", agg)
            all_summaries.append(agg)
            benchmark_summary["results"][protocol][model_name] = agg
            print(json.dumps(agg, indent=2))

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(out / "model_comparison.csv", index=False)

    # Primary model is selected by blocked-CV realized gain only for diagnostic reporting.
    primary_protocol = "blocked" if "blocked" in protocols else protocols[0]
    primary = summary_df[summary_df.protocol == primary_protocol].copy()
    best = None
    if not primary.empty:
        best = primary.sort_values("realized_gain_pct", ascending=False).iloc[0].to_dict()
    benchmark_summary["primary_protocol"] = primary_protocol
    benchmark_summary["best_realized_model"] = best
    if best:
        g = float(best["realized_gain_pct"])
        if g >= 2.0:
            guidance = "WINDOW_LEVEL_REALIZABLE__PROCEED_TO_PREDICTIVE_UTILITY_SELECTOR"
        elif g >= 1.0:
            guidance = "PARTIAL_REALIZABILITY__IMPROVE_FEATURES_AND_COMPARE_LOCAL_MEMORY"
        else:
            guidance = "LOW_REALIZABILITY__PRIORITIZE_LOCAL_STATE_REPRESENTATION_OR_MEMORY"
    else:
        guidance = "NO_RESULT"
    benchmark_summary["diagnostic_guidance"] = guidance
    save_json(out / "summary.json", benchmark_summary)

    print("\n" + "=" * 92)
    print("SRD v0.4 REALIZABLE BENCHMARK COMPLETE")
    print("primary_protocol:", primary_protocol)
    print("best_realized_model:", json.dumps(best, indent=2, default=str) if best else None)
    print("guidance:", guidance)
    print("model_comparison:", out / "model_comparison.csv")
    print("summary:", out / "summary.json")
    print("=" * 92)


if __name__ == "__main__":
    main()
