#!/usr/bin/env python3
"""Measure LEAF Graph/Hyper selector opportunity on Val labels.

This is a diagnostic upper bound only.  It deliberately uses validation future
labels to select the better branch for each (query, node), matching the current
LEAF selector granularity.  It must never be reported as a deployable method or
used on Test before final evaluation authorization.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path("/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline")
ROOT = Path(os.environ.get("BASELINE_ROOT", str(DEFAULT_ROOT)))
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.analysis.leaf_selector_headroom import (
    apply_query_node_choices,
    build_query_node_oracle,
    choice_statistics,
    relative_gain,
)
from baseline.metrics import metric_bundle
from baseline.runners.common import validate_stage
from baseline.utils import write_json


def latest_branch_run(cfg: dict, stage: str) -> Path:
    ptr = (
        Path(cfg["project"]["exp_root"])
        / "LEAF"
        / cfg["experiment"]["subgraph"]
        / stage
        / "LATEST_BRANCH_RUN.txt"
    )
    if not ptr.exists():
        raise RuntimeError(f"LEAF branch pointer not found: {ptr}")
    run = Path(ptr.read_text().strip())
    if not run.exists():
        raise RuntimeError(f"Stale LEAF branch pointer: {run}")
    return run


def load_branch_array(run: Path, name: str) -> np.ndarray:
    path = run / name
    if not path.exists():
        raise RuntimeError(f"Required LEAF branch artifact missing: {path}")
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 3:
        raise RuntimeError(f"Unexpected {name} shape {arr.shape}; expected [B,P,N]")
    return arr


def all_metrics(pred, target, valid, direct, support) -> dict:
    return {
        k: (v.item() if hasattr(v, "item") else v)
        for k, v in metric_bundle(pred, target, valid, direct, support).items()
    }


def metric_rows(graph_m: dict, hyper_m: dict, oracle_m: dict) -> pd.DataFrame:
    rows = []
    for name, m in (("GraphBranch", graph_m), ("HypergraphBranch", hyper_m), ("QueryNodeOracle", oracle_m)):
        rows.append({"variant": name, **m})
    return pd.DataFrame(rows)


def headroom_class(gain: float) -> str:
    if gain >= 0.02:
        return "HIGH_gt_2pct"
    if gain >= 0.005:
        return "MODERATE_0.5_to_2pct"
    return "LOW_lt_0.5pct"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["development", "formal"], default="development")
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--branch-run", default=None, help="Explicit LEAF-branches run directory; default uses LATEST_BRANCH_RUN.txt")
    ap.add_argument("--output-dir", default=None, help="Default: <branch-run>/selector_headroom")
    args = ap.parse_args()

    cfg, cache = validate_stage(args.stage, "LEAF")
    run = Path(args.branch_run).resolve() if args.branch_run else latest_branch_run(cfg, args.stage)
    if not run.exists():
        raise RuntimeError(f"LEAF branch run does not exist: {run}")

    summary_path = run / "summary.json"
    branch_summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if branch_summary:
        if str(branch_summary.get("stage")) != args.stage:
            raise RuntimeError(f"Branch stage mismatch: summary={branch_summary.get('stage')} requested={args.stage}")
        if int(branch_summary.get("seed", args.seed)) != int(args.seed):
            raise RuntimeError(f"Branch seed mismatch: summary={branch_summary.get('seed')} requested={args.seed}")

    graph = load_branch_array(run, "graph_val_raw.npy")
    hyper = load_branch_array(run, "hyper_val_raw.npy")
    target = np.load(cache / "val_y_raw.npy", mmap_mode="r")
    valid = np.load(cache / "val_future_valid.npy", mmap_mode="r")
    direct = np.load(cache / "val_direct_mask.npy", mmap_mode="r")
    support = np.load(cache / "val_support_mask.npy", mmap_mode="r")
    if tuple(graph.shape) != tuple(target.shape) or tuple(hyper.shape) != tuple(target.shape):
        raise RuntimeError(
            f"Prediction/target mismatch graph={graph.shape} hyper={hyper.shape} target={target.shape}"
        )

    out = Path(args.output_dir).resolve() if args.output_dir else run / "selector_headroom"
    out.mkdir(parents=True, exist_ok=True)

    oracle = build_query_node_oracle(graph, hyper, target, valid, tie_break="hyper")
    np.save(out / "oracle_query_node_choices.npy", oracle.choices)
    np.save(out / "oracle_query_node_eligible.npy", oracle.eligible.astype(np.uint8))

    oracle_pred = apply_query_node_choices(
        graph,
        hyper,
        oracle.choices,
        out_path=out / "oracle_val_raw.npy",
    )

    graph_m = all_metrics(graph, target, valid, direct, support)
    hyper_m = all_metrics(hyper, target, valid, direct, support)
    oracle_m = all_metrics(oracle_pred, target, valid, direct, support)
    metrics_df = metric_rows(graph_m, hyper_m, oracle_m)
    metrics_df.to_csv(out / "selector_headroom_metrics.csv", index=False)

    stats = choice_statistics(oracle)
    tracked = [
        "overall_MAE",
        "masked_MAE",
        "support_MAE",
        "H3_overall_MAE",
        "H3_masked_MAE",
        "H3_support_MAE",
        "H6_overall_MAE",
        "H6_masked_MAE",
        "H6_support_MAE",
        "H12_overall_MAE",
        "H12_masked_MAE",
        "H12_support_MAE",
    ]
    gains = {
        key: relative_gain(hyper_m[key], oracle_m[key])
        for key in tracked
        if key in hyper_m and key in oracle_m
    }
    overall_gain = float(gains.get("overall_MAE", float("nan")))

    result = {
        "artifact_type": "leaf_selector_headroom_diagnostic",
        "revision": "leaf_selector_headroom_r1",
        "stage": args.stage,
        "seed": int(args.seed),
        "branch_run": str(run),
        "cache": str(cache),
        "selection_granularity": "query_node_one_branch_for_all_12_horizons",
        "selection_objective": "lowest_valid_future_12step_MAE_per_query_node",
        "tie_break": "HypergraphBranch",
        "uses_validation_future_labels": True,
        "diagnostic_only": True,
        "deployable": False,
        "test_access": False,
        "graph": graph_m,
        "hyper": hyper_m,
        "oracle": oracle_m,
        "oracle_relative_gain_over_always_hyper": gains,
        "overall_MAE_headroom_class": headroom_class(overall_gain),
        "choice_statistics": stats,
        "decision_rule": {
            "LOW_lt_0.5pct": "Selector opportunity is weak; keep LEAF mainly as a branch baseline.",
            "MODERATE_0.5_to_2pct": "Run Qwen selectors, but expect limited aggregate gain.",
            "HIGH_gt_2pct": "Candidate complementarity is meaningful; prioritize Qwen selector experiments.",
        },
        "artifacts": {
            "choices": str(out / "oracle_query_node_choices.npy"),
            "eligible": str(out / "oracle_query_node_eligible.npy"),
            "oracle_prediction": str(out / "oracle_val_raw.npy"),
            "metrics_csv": str(out / "selector_headroom_metrics.csv"),
        },
    }
    write_json(out / "selector_headroom_summary.json", result)

    print("LEAF SELECTOR HEADROOM DIAGNOSTIC")
    print(f"branch_run={run}")
    print(f"Graph Overall/Masked/Support MAE = {graph_m['overall_MAE']:.6f} / {graph_m['masked_MAE']:.6f} / {graph_m['support_MAE']:.6f}")
    print(f"Hyper Overall/Masked/Support MAE = {hyper_m['overall_MAE']:.6f} / {hyper_m['masked_MAE']:.6f} / {hyper_m['support_MAE']:.6f}")
    print(f"Oracle Overall/Masked/Support MAE = {oracle_m['overall_MAE']:.6f} / {oracle_m['masked_MAE']:.6f} / {oracle_m['support_MAE']:.6f}")
    print(
        "Oracle relative gain over always-Hyper = "
        f"Overall {100*gains['overall_MAE']:.3f}% | "
        f"Masked {100*gains['masked_MAE']:.3f}% | "
        f"Support {100*gains['support_MAE']:.3f}%"
    )
    print(
        "Strict branch wins = "
        f"Graph {100*stats['graph_strict_win_fraction']:.2f}% | "
        f"Hyper {100*stats['hyper_strict_win_fraction']:.2f}% | "
        f"Tie {100*stats['tie_fraction']:.2f}%"
    )
    print(f"HEADROOM_CLASS={result['overall_MAE_headroom_class']}")
    print(f"wrote {out / 'selector_headroom_summary.json'}")
    print("LEAF SELECTOR HEADROOM PASS")


if __name__ == "__main__":
    main()
