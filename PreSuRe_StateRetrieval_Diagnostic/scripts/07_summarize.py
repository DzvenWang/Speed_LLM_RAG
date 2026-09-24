#!/usr/bin/env python3
from pathlib import Path
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    a = ap.parse_args()
    root = Path(a.run_dir)
    files = sorted(root.rglob("summary.json"))
    if not files:
        raise SystemExit(f"No summary.json under {root}")

    print("=" * 92)
    print("PreSuRe State-Retrieval Diagnostic v0.3 summaries")
    print("=" * 92)

    ceiling = root / "window_ceiling_sweep" / "summary.json"
    if ceiling.exists():
        c = json.loads(ceiling.read_text(encoding="utf-8"))
        print("\n[PRIMARY WINDOW-LEVEL CEILING DIAGNOSTIC]")
        print("transport:", c.get("transport"))
        print("recent_steps:", c.get("recent_steps"))
        print("k_values:", c.get("k_values"))
        print("best_fixed_scope_at_max_k:", json.dumps(c.get("best_fixed_scope_at_max_k", {}), indent=2))
        print("cross_scope_at_max_k:", json.dumps(c.get("cross_scope_at_max_k", {}), indent=2))
        print("scope_choice_counts:", json.dumps(c.get("scope_choice_counts", {}), indent=2))
        print("diagnostic_guidance:", c.get("diagnostic_guidance"))
        matrix = root / "window_ceiling_sweep" / "oracle_gain_matrix.csv"
        if matrix.exists():
            print("oracle_gain_matrix:", matrix)
        cross = root / "window_ceiling_sweep" / "cross_scope_oracle.csv"
        if cross.exists():
            print("cross_scope_oracle:", cross)

    for p in files:
        if p == ceiling:
            continue
        print(f"\n[{p.relative_to(root)}]")
        j = json.loads(p.read_text(encoding="utf-8"))
        for k in [
            "baseline_mean_per_query_masked_mae",
            "mean_top1_observed_mae",
            "mean_top1_observed_mae_recent3",
            "mean_top1_observed_mae_recent6",
            "beneficial_rate",
            "harmful_rate",
            "mean_utility",
        ]:
            if k in j:
                print(f"{k}: {j[k]}")
        if "suffixes" in j:
            print("suffixes:", json.dumps(j["suffixes"], indent=2))
        if "similarity_top1" in j:
            print("similarity_top1:", json.dumps(j["similarity_top1"], indent=2))
        if "oracle_at_k" in j:
            print("oracle_at_k (with no-intervention):", json.dumps(j["oracle_at_k"], indent=2))


if __name__ == "__main__":
    main()
