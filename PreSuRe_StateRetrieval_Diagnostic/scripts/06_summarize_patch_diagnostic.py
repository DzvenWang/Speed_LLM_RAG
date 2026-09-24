#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from srd.utils import write_json


def _read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _best_mode(table, mode):
    z = table[table["mode"] == mode]
    return float(z.forecast_gain_pct.max()) if len(z) else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Summarize Patch-Level Forecast-Utility Diagnostic v0.6")
    ap.add_argument("run_dir")
    a = ap.parse_args()
    run = Path(a.run_dir).resolve()
    required = {
        "layout": run / "patch_layout" / "patch_layout.json",
        "restoration": run / "patch_restoration_oracle" / "summary.json",
        "retrieval": run / "patch_retrieval" / "summary.json",
        "completion": next(iter(sorted(run.glob("completion_oracle_*/summary.json"))), None),
        "realizable": run / "transport_realizable" / "summary.json",
        "forecast_utility": run / "forecast_utility_oracle" / "summary.json",
    }
    missing = [k for k, p in required.items() if p is None or not Path(p).exists()]
    if missing:
        raise RuntimeError(f"Missing diagnostic outputs: {missing}; run={run}")

    layout = _read_json(required["layout"])
    restoration = _read_json(required["restoration"])
    retrieval = _read_json(required["retrieval"])
    completion = _read_json(required["completion"])
    realizable = _read_json(required["realizable"])
    forecast = _read_json(required["forecast_utility"])

    oracle_table = pd.read_csv(Path(required["completion"]).parent / "oracle_by_k.csv")
    real_table = pd.read_csv(run / "transport_realizable" / "summary_table.csv")
    fut_table = pd.read_csv(run / "forecast_utility_oracle" / "forecast_oracle_by_k.csv")

    patch_scopes = restoration.get("scopes", {})
    best_true = max((float(v.get("all_damaged_patches_gain_pct", float("nan"))) for v in patch_scopes.values()), default=float("nan"))
    best_completion = float(oracle_table.forecast_gain_pct.max()) if len(oracle_table) else float("nan")
    top1 = real_table[real_table["mode"] == "top1"]
    weighted = real_table[real_table["mode"] == "weighted_topk"]
    best_top1 = float(top1.forecast_gain_pct.max()) if len(top1) else float("nan")
    best_weighted = float(weighted.forecast_gain_pct.max()) if len(weighted) else float("nan")
    best_realizable = max(best_top1, best_weighted)

    best_single = _best_mode(fut_table, "single_patch_forecast_oracle")
    best_all_beneficial = _best_mode(fut_table, "all_individually_beneficial_patches")
    best_subset = _best_mode(fut_table, "patch_subset_prefix_forecast_oracle")
    best_coherent = _best_mode(fut_table, "same_episode_coherent_forecast_oracle")
    forecast_oracle_values = [x for x in [best_single, best_subset, best_coherent, best_all_beneficial] if x == x]
    best_forecast_oracle = max(forecast_oracle_values) if forecast_oracle_values else float("nan")
    completion_to_forecast_gap = best_forecast_oracle - best_completion if best_forecast_oracle == best_forecast_oracle and best_completion == best_completion else float("nan")
    realizable_conversion = (best_realizable / best_forecast_oracle) if best_forecast_oracle and best_forecast_oracle == best_forecast_oracle else float("nan")

    cfg_target = 8.0
    strong_target = 10.0
    try:
        cfg_target = float(forecast.get("candidate_space_target_pct", cfg_target))
        strong_target = float(forecast.get("strong_candidate_space_target_pct", strong_target))
    except Exception:
        pass

    if best_forecast_oracle == best_forecast_oracle and best_forecast_oracle < 2.0:
        route_signal = "candidate_or_input_injection_limited"
    elif best_coherent == best_coherent and best_subset == best_subset and best_coherent >= best_subset + 1.0:
        route_signal = "cross_patch_coherence_limited"
    elif best_single == best_single and best_subset == best_subset and best_single >= 2.0 and best_subset < 0.7 * best_single:
        route_signal = "patch_interaction_limited"
    elif best_forecast_oracle == best_forecast_oracle and best_completion == best_completion and best_forecast_oracle >= 4.0 and best_completion <= 1.5:
        route_signal = "completion_objective_mismatch"
    else:
        route_signal = "mixed_diagnostic_signal"

    verdict = {
        "true_state_completion_headroom_pct": best_true,
        "hidden_history_completion_oracle_pct": best_completion,
        "single_patch_forecast_oracle_pct": best_single,
        "all_individually_beneficial_patches_pct": best_all_beneficial,
        "patch_subset_prefix_forecast_oracle_pct": best_subset,
        "same_episode_coherent_forecast_oracle_pct": best_coherent,
        "best_tested_forecast_utility_oracle_pct": best_forecast_oracle,
        "best_realizable_patch_retrieval_pct": best_realizable,
        "completion_to_forecast_oracle_gap_pct": completion_to_forecast_gap,
        "realizable_to_forecast_oracle_ratio": realizable_conversion,
        "tested_candidate_space_reaches_8pct": bool(best_forecast_oracle >= cfg_target),
        "tested_candidate_space_reaches_10pct": bool(best_forecast_oracle >= strong_target),
        "realizable_reaches_2pct": bool(best_realizable >= 2.0),
        "route_signal": route_signal,
    }
    payload = {
        "version": "0.6.1",
        "run_dir": str(run),
        "patch_layout": {
            "patch_count": layout["patch_count"], "patch_size_slots": layout["patch_size_slots"],
            "node_num": layout["node_num"], "source": layout["source"],
            "node_counts": layout.get("metadata", {}).get("node_counts"),
        },
        "retrieval": retrieval,
        "forecast_utility_diagnostic": {k: v for k, v in forecast.items() if k != "results"},
        "verdict": verdict,
    }
    write_json(run / "patch_diagnostic_summary.json", payload)

    lines = [
        "# Patch-Level Forecast-Utility Diagnostic v0.6.1", "",
        "## Core numbers", "",
        f"- PatchSTG spatial patches recovered: **{layout['patch_count']}** over {layout['node_num']} nodes.",
        f"- True-state structured restoration headroom: **{best_true:.3f}%**.",
        f"- Hidden-history completion Oracle downstream gain: **{best_completion:.3f}%**.",
        f"- Single-patch Forecast Oracle gain: **{best_single:.3f}%**.",
        f"- Utility-ranked patch-subset Forecast Oracle gain: **{best_subset:.3f}%**.",
        f"- Same-episode coherent Forecast Oracle gain: **{best_coherent:.3f}%**.",
        f"- Best tested Forecast-Utility Oracle gain: **{best_forecast_oracle:.3f}%**.",
        f"- Best inference-available retrieval gain: **{best_realizable:.3f}%**.",
        "", "## Decision checks", "",
        f"- Tested forecast-utility candidate/action space reaches 8%: **{verdict['tested_candidate_space_reaches_8pct']}**.",
        f"- Tested forecast-utility candidate/action space reaches 10%: **{verdict['tested_candidate_space_reaches_10pct']}**.",
        f"- Realizable retrieval reaches 2%: **{verdict['realizable_reaches_2pct']}**.",
        f"- Diagnostic route signal: **{route_signal}**.",
        "", "## Interpretation rules", "",
        "1. The hidden-history completion Oracle uses current-history truth only and never uses future targets.",
        "2. Every Forecast Oracle in v0.6 first retrieves candidates using inference-available observed history, then uses the true future target only to measure/choose diagnostic actions. It is an upper-bound diagnostic, not an inference method.",
        "3. `single_patch_forecast_oracle` tests whether any local retrieved support is useful by itself.",
        "4. `patch_subset_prefix_forecast_oracle` tests interaction among individually beneficial patch actions.",
        "5. `same_episode_coherent_forecast_oracle` forces all damaged patches to use one historical episode and tests whether cross-patch coherence is the limiting factor.",
    ]
    (run / "PATCH_DIAGNOSTIC_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"SUMMARY={run / 'patch_diagnostic_summary.json'}")


if __name__ == "__main__":
    main()
