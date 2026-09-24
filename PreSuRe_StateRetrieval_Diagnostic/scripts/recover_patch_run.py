#!/usr/bin/env python3
"""Recover a v0.5.x patch diagnostic that finished Stage 05 computation
but failed while writing summaries.

This script never reruns PatchSTG inference. It reconstructs
transport_realizable/summary.json from the already-written CSV table and
then runs the normal Stage 06 summarizer.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from srd.config import load_config
from srd.utils import write_json


def _finite_max(series):
    s = pd.to_numeric(series, errors="coerce")
    return float(s.max()) if s.notna().any() else float("nan")


def main():
    ap = argparse.ArgumentParser(description="Recover an already-computed Patch Diagnostic v0.5.x run without rerunning GPU inference")
    ap.add_argument("--stage", default="development", choices=["development", "formal"])
    ap.add_argument("--run-tag", default=None, help="Existing run tag, e.g. patchdiag_v051.20260923_205431")
    ap.add_argument("--run-dir", default=None, help="Explicit existing experiment run directory")
    a = ap.parse_args()

    cfg = load_config(ROOT)
    dataset = cfg["experiment"]["subgraph"]
    if a.run_dir:
        run = Path(a.run_dir).expanduser().resolve()
        run_tag = run.name
    else:
        if not a.run_tag:
            ap.error("one of --run-tag or --run-dir is required")
        run_tag = a.run_tag
        run = Path(cfg["project"]["exp_root"]) / dataset / a.stage / run_tag
    tr = run / "transport_realizable"
    table_path = tr / "summary_table.csv"
    per_query_path = tr / "per_query.csv"
    if not table_path.exists():
        raise FileNotFoundError(
            f"Stage 05 summary table not found: {table_path}\n"
            "The failed run must have reached the point where summary_table.csv was written."
        )
    if not per_query_path.exists():
        raise FileNotFoundError(f"Stage 05 per-query table not found: {per_query_path}")

    sm = pd.read_csv(table_path)
    required_cols = {"mode", "forecast_gain_pct"}
    missing = required_cols.difference(sm.columns)
    if missing:
        raise RuntimeError(f"summary_table.csv missing columns: {sorted(missing)}")

    top1 = sm[sm["mode"] == "top1"]
    weighted = sm[sm["mode"] == "weighted_topk"]
    oracle = sm[sm["mode"] == "completion_oracle"]
    best_top1 = _finite_max(top1["forecast_gain_pct"]) if len(top1) else float("nan")
    best_weighted = _finite_max(weighted["forecast_gain_pct"]) if len(weighted) else float("nan")
    best_oracle = _finite_max(oracle["forecast_gain_pct"]) if len(oracle) else float("nan")

    payload = {
        "version": "0.5.2",
        "recovered_from_existing_stage05_tables": True,
        "run_tag": run_tag,
        "queries": int(pd.read_csv(per_query_path, usecols=["query_id"])["query_id"].nunique()),
        "top1_uses_hidden_truth": False,
        "completion_oracle_uses_hidden_history_truth": True,
        "completion_oracle_uses_future_forecast_target": False,
        "best_top1_gain_pct": best_top1,
        "best_weighted_topk_gain_pct": best_weighted,
        "best_completion_oracle_gain_pct": best_oracle,
        "results": sm.to_dict(orient="records"),
    }
    write_json(tr / "summary.json", payload)
    print(f"RECOVERED_STAGE05_SUMMARY={tr / 'summary.json'}")

    summary_script = ROOT / "scripts" / "06_summarize_patch_diagnostic.py"
    cmd = [sys.executable, str(summary_script), str(run)]
    print(">> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print("PATCH_DIAGNOSTIC_RECOVERY_COMPLETE")


if __name__ == "__main__":
    main()
