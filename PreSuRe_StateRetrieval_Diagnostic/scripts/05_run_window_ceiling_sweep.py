#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._common import setup, run_dir
from srd.utils import write_json


def _parse_int_list(text):
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="SRD v0.3 raw-copy window-level ceiling sweep")
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--stage", default="development", choices=["development", "formal"])
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--max-k", type=int, default=50)
    ap.add_argument("--recent-steps", default=None,
                    help="Comma-separated restoration scopes, e.g. 1,2,3,4,6. Defaults to config.")
    a = ap.parse_args()

    cfg = setup(a.stage)
    sweep_cfg = cfg.get("ceiling_sweep", {})
    retrieval_npz = np.load(a.retrieval)
    available_k = int(retrieval_npz["candidate_ids"].shape[1])
    effective_max_k = min(int(a.max_k), available_k)
    scopes = _parse_int_list(a.recent_steps) if a.recent_steps else [int(x) for x in sweep_cfg.get("recent_steps", [1,2,3,4,6])]
    scopes = sorted(set(max(1, min(int(cfg["experiment"]["history"]), int(x))) for x in scopes))
    transport = str(sweep_cfg.get("transport", "raw_copy"))
    configured_k = [int(x) for x in sweep_cfg.get("k_values", cfg["oracle"].get("k_values", [1,5,10,20,50]))]
    k_values = sorted(set(k for k in configured_k if k <= effective_max_k))
    if effective_max_k not in k_values:
        k_values.append(effective_max_k)
    k_values = sorted(set(k_values))

    script = ROOT / "scripts" / "04_run_retrieval_oracle.py"
    base = run_dir(cfg, a.stage, a.run_tag)
    rows = []
    summaries = {}

    for L in scopes:
        cmd = [
            sys.executable, str(script),
            "--retrieval", a.retrieval,
            "--transport", transport,
            "--recent-steps", str(L),
            "--stage", a.stage,
            "--run-tag", a.run_tag,
            "--max-k", str(effective_max_k),
        ]
        if a.checkpoint:
            cmd += ["--checkpoint", a.checkpoint]
        print("RUN", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        scope_name = f"recent{L}"
        sp = base / f"oracle_{transport}_{scope_name}" / "summary.json"
        s = json.loads(sp.read_text(encoding="utf-8"))
        summaries[L] = s
        for k in k_values:
            key = str(k)
            if key not in s["oracle_at_k"]:
                continue
            o = s["oracle_at_k"][key]
            forced = s["candidate_only_oracle_at_k"][key]
            rows.append({
                "recent_steps": L,
                "scope": scope_name,
                "transport": transport,
                "k": k,
                "baseline_mae": s["baseline_mean_per_query_masked_mae"],
                "top1_mae": s["similarity_top1"]["mean_per_query_masked_mae"],
                "top1_gain_pct": s["similarity_top1"]["gain_pct"],
                "top1_beneficial_rate": s["similarity_top1"]["beneficial_query_rate"],
                "top1_harmful_rate": s["similarity_top1"]["harmful_query_rate"],
                "oracle_mae": o["mean_per_query_masked_mae"],
                "oracle_gain_pct": o["gain_pct"],
                "candidate_hit_rate": o["candidate_hit_rate"],
                "no_intervention_rate": o["no_intervention_rate"],
                "forced_oracle_mae": forced["mean_per_query_masked_mae"],
                "forced_oracle_gain_pct": forced["gain_pct"],
            })

    out = base / "window_ceiling_sweep"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["recent_steps", "k"])
    df.to_csv(out / "ceiling_long.csv", index=False)

    gain_matrix = df.pivot(index="recent_steps", columns="k", values="oracle_gain_pct").sort_index()
    hit_matrix = df.pivot(index="recent_steps", columns="k", values="candidate_hit_rate").sort_index()
    gain_matrix.to_csv(out / "oracle_gain_matrix.csv")
    hit_matrix.to_csv(out / "candidate_hit_matrix.csv")

    # Cross-scope oracle: for a fixed K, let each query choose both the candidate
    # and restoration suffix. This is a diagnostic upper bound for future scope selection.
    evals = {}
    common_qids = None
    common_base = None
    for L in scopes:
        ep = base / f"oracle_{transport}_recent{L}" / "candidate_eval.npz"
        z = np.load(ep)
        qids = z["query_ids"].astype(np.int64)
        base_mae = z["baseline_mae"].astype(np.float64)
        cand_mae = z["candidate_mae"].astype(np.float64)
        if common_qids is None:
            common_qids = qids
            common_base = base_mae
        else:
            if not np.array_equal(common_qids, qids):
                raise RuntimeError(f"Query IDs differ across scope sweep at recent{L}")
            if not np.allclose(common_base, base_mae, equal_nan=True, atol=1e-6):
                raise RuntimeError(f"Baseline MAE differs across scope sweep at recent{L}")
        evals[L] = cand_mae

    cross_rows = []
    scope_choice = {}
    bmean = float(np.nanmean(common_base))
    for k in k_values:
        per_scope_best = []
        for L in scopes:
            arr = evals[L][:, : min(k, evals[L].shape[1])]
            per_scope_best.append(np.nanmin(arr, axis=1))
        stacked = np.stack(per_scope_best, axis=1)  # [Q, scopes]
        safe_stacked = np.where(np.isfinite(stacked), stacked, np.inf)
        best_scope_idx = np.argmin(safe_stacked, axis=1)
        best_candidate = np.min(safe_stacked, axis=1)
        safe = np.minimum(common_base, best_candidate)
        used = np.isfinite(best_candidate) & (best_candidate < common_base)
        safe_mean = float(np.nanmean(safe))
        gain = (bmean - safe_mean) / bmean * 100.0 if bmean != 0 else float("nan")
        counts = {str(L): 0 for L in scopes}
        for idx, flag in zip(best_scope_idx, used):
            if flag:
                counts[str(scopes[int(idx)])] += 1
        counts["no_intervention"] = int((~used).sum())
        scope_choice[str(k)] = counts
        cross_rows.append({
            "k": k,
            "baseline_mae": bmean,
            "cross_scope_oracle_mae": safe_mean,
            "cross_scope_oracle_gain_pct": gain,
            "candidate_hit_rate": float(np.mean(used)),
            "no_intervention_rate": float(np.mean(~used)),
        })

    cross_df = pd.DataFrame(cross_rows).sort_values("k")
    cross_df.to_csv(out / "cross_scope_oracle.csv", index=False)

    # Compact decision-oriented summary. Thresholds are guidance only.
    max_k = max(k_values)
    at_max = df[df["k"] == max_k]
    best_row = at_max.loc[at_max["oracle_gain_pct"].idxmax()].to_dict()
    cross_max = cross_df[cross_df["k"] == max_k].iloc[0].to_dict()
    best_fixed_gain = float(best_row["oracle_gain_pct"])
    cross_gain = float(cross_max["cross_scope_oracle_gain_pct"])
    rerank_thr = float(sweep_cfg.get("utility_reranker_threshold_pct", 4.0))
    local_thr = float(sweep_cfg.get("local_memory_threshold_pct", 3.0))
    if best_fixed_gain >= rerank_thr:
        guidance = "WINDOW_MEMORY_HAS_HEADROOM__NEXT_TEST_UTILITY_RERANKER"
    elif best_fixed_gain < local_thr:
        guidance = "WINDOW_MEMORY_LOW_CEILING__NEXT_TEST_LOCAL_NODE_MEMORY"
    else:
        guidance = "BORDERLINE__EXPAND_MEMORY_OR_COMPARE_LOCAL_MEMORY_BEFORE_RERANKER"

    summary = {
        "version": "0.3.0",
        "transport": transport,
        "recent_steps": scopes,
        "k_values": k_values,
        "retrieval_available_k": available_k,
        "effective_max_k": effective_max_k,
        "best_fixed_scope_at_max_k": {
            "k": max_k,
            "recent_steps": int(best_row["recent_steps"]),
            "oracle_gain_pct": best_fixed_gain,
            "oracle_mae": float(best_row["oracle_mae"]),
            "candidate_hit_rate": float(best_row["candidate_hit_rate"]),
        },
        "cross_scope_at_max_k": {
            "k": max_k,
            "oracle_gain_pct": cross_gain,
            "oracle_mae": float(cross_max["cross_scope_oracle_mae"]),
            "candidate_hit_rate": float(cross_max["candidate_hit_rate"]),
            "gain_over_best_fixed_scope_pct_points": cross_gain - best_fixed_gain,
        },
        "scope_choice_counts": scope_choice,
        "diagnostic_guidance": guidance,
        "thresholds_pct": {
            "utility_reranker": rerank_thr,
            "local_memory": local_thr,
        },
        "files": {
            "ceiling_long": str((out / "ceiling_long.csv").resolve()),
            "oracle_gain_matrix": str((out / "oracle_gain_matrix.csv").resolve()),
            "candidate_hit_matrix": str((out / "candidate_hit_matrix.csv").resolve()),
            "cross_scope_oracle": str((out / "cross_scope_oracle.csv").resolve()),
        },
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print("WINDOW RAW-COPY CEILING SWEEP PASS")


if __name__ == "__main__":
    main()
