#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._common import setup, run_dir, default_tag
from srd.contract import assert_cache_ready
from srd.backbone import load_patchstg_adapter
from srd.data import CacheView
from srd.evaluate import predict_raw_batched
from srd.metrics import masked_mae, improvement_pct
from srd.patch_layout import extract_patch_layout
from srd.patch_transport import suffix_mask
from srd.utils import deterministic_ids, write_json


def _mae_for_variants(adapter, ck, q, variants, device, batch_size):
    X = np.stack(variants).astype(np.float32)
    tod = np.repeat(q["tod_hist"][None, :], len(X), axis=0)
    dow = np.repeat(q["dow_hist"][None, :], len(X), axis=0)
    pred = predict_raw_batched(adapter, ck, X, tod, dow, device, batch_size=batch_size)
    y = np.repeat(q["y_raw"][None, :, :], len(X), axis=0)
    fv = np.repeat(q["future_valid"][None, :, :], len(X), axis=0)
    dm = np.repeat(q["direct_mask"][None, :], len(X), axis=0)
    return masked_mae(pred, y, fv, dm)


def main():
    ap = argparse.ArgumentParser(description="PatchSTG patch-level true-state restoration oracle")
    ap.add_argument("--stage", default="development", choices=["development", "formal"])
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--patch-size", type=int, default=None, help="Explicit PatchSTG spatial patch width in reordered slots (debug override)")
    ap.add_argument("--patch-count", type=int, default=None, help="Explicit number of PatchSTG spatial patches (debug override)")
    ap.add_argument("--queries", type=int, default=None)
    ap.add_argument("--scopes", default=None, help="Comma-separated recent suffixes, default config patch_level.suffix_lengths")
    ap.add_argument("--eval-batch-size", type=int, default=64)
    a = ap.parse_args()

    cfg = setup(a.stage)
    assert_cache_ready(cfg, a.stage)
    tag = a.run_tag or default_tag("patchdiag_v060")
    out = run_dir(cfg, a.stage, tag) / "patch_restoration_oracle"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adapter, cp, ck, source = load_patchstg_adapter(cfg, a.stage, device, a.checkpoint)
    layout = extract_patch_layout(adapter.model, ck, backbone_root=cfg["project"]["backbone_root"], patch_size_override=a.patch_size, patch_count_override=a.patch_count)
    val = CacheView(cfg, a.stage, "val")
    ids = deterministic_ids(len(val), a.queries or cfg["experiment"]["diagnostic_queries"], cfg["experiment"]["seed"])
    H = int(cfg["experiment"]["history"])
    scopes = [int(x) for x in (a.scopes.split(",") if a.scopes else cfg["patch_level"]["suffix_lengths"])]
    scopes = sorted(set(max(1, min(H, x)) for x in scopes))

    action_rows, query_rows = [], []
    for z, qi in enumerate(ids, 1):
        q = val.row(int(qi))
        variants = [q["x_obs"].copy()]
        labels = [("baseline", None, None, 0)]
        for L in scopes:
            scope = suffix_mask(q["restorable"].shape, L)
            # All damaged patches restored together: should agree with recent-suffix oracle.
            xa = q["x_obs"].copy()
            all_fill = q["restorable"] & scope
            xa[all_fill] = q["x_full"][all_fill]
            variants.append(xa)
            labels.append(("all_patches", L, None, int(all_fill.sum())))

            for pid, nodes in enumerate(layout.groups):
                fill = np.zeros_like(q["restorable"], dtype=bool)
                fill[:, nodes] = True
                fill &= q["restorable"] & scope
                if not np.any(fill):
                    continue
                xp = q["x_obs"].copy()
                xp[fill] = q["x_full"][fill]
                variants.append(xp)
                labels.append(("single_patch", L, pid, int(fill.sum())))

        maes = _mae_for_variants(adapter, ck, q, variants, device, a.eval_batch_size)
        base = float(maes[0])
        per_scope_actions = {L: [] for L in scopes}
        all_patch_mae = {}
        for j, (kind, L, pid, fill_count) in enumerate(labels[1:], 1):
            model_mae = float(maes[j])
            utility = base - model_mae
            if kind == "all_patches":
                all_patch_mae[int(L)] = model_mae
            else:
                per_scope_actions[int(L)].append(utility)
                action_rows.append({
                    "query_id": int(qi),
                    "patch_id": int(pid),
                    "recent_steps": int(L),
                    "fill_count": int(fill_count),
                    "baseline_mae": base,
                    "restored_mae": model_mae,
                    "utility_mae": utility,
                    "gain_pct": improvement_pct(base, model_mae),
                    "beneficial": bool(utility > 0),
                })
        row = {
            "query_id": int(qi),
            "baseline_mae": base,
            "missing_ratio": float(q["restorable"].sum() / max(q["full_valid"].sum(), 1)),
            "damaged_patch_count": int(sum(np.any(q["restorable"][:, nodes]) for nodes in layout.groups)),
        }
        for L in scopes:
            am = float(all_patch_mae[L])
            actions = per_scope_actions[L]
            best_utility = max([0.0, *actions]) if actions else 0.0
            row[f"recent{L}_all_patch_mae"] = am
            row[f"recent{L}_all_patch_gain_pct"] = improvement_pct(base, am)
            row[f"recent{L}_best_single_patch_mae"] = base - best_utility
            row[f"recent{L}_best_single_patch_gain_pct"] = improvement_pct(base, base - best_utility)
        query_rows.append(row)
        if z % 10 == 0 or z == len(ids):
            print(f"PATCH RESTORATION ORACLE {z}/{len(ids)}", flush=True)

    adf = pd.DataFrame(action_rows)
    qdf = pd.DataFrame(query_rows)
    adf.to_csv(out / "single_patch_actions.csv", index=False)
    qdf.to_csv(out / "per_query_scope.csv", index=False)
    baseline = float(qdf["baseline_mae"].mean())
    summary = {
        "queries": int(len(qdf)),
        "patch_count": int(layout.patch_count),
        "patch_node_counts": [int(len(x)) for x in layout.groups],
        "checkpoint": str(cp),
        "checkpoint_source": source,
        "baseline_mean_per_query_masked_mae": baseline,
        "scopes": {},
    }
    for L in scopes:
        allm = float(qdf[f"recent{L}_all_patch_mae"].mean())
        bestm = float(qdf[f"recent{L}_best_single_patch_mae"].mean())
        sdf = adf[adf.recent_steps == L] if not adf.empty else adf
        summary["scopes"][str(L)] = {
            "all_damaged_patches_true_restoration_mae": allm,
            "all_damaged_patches_gain_pct": improvement_pct(baseline, allm),
            "best_single_patch_true_restoration_mae": bestm,
            "best_single_patch_gain_pct": improvement_pct(baseline, bestm),
            "single_patch_beneficial_action_rate": float(sdf.beneficial.mean()) if len(sdf) else float("nan"),
            "single_patch_action_count": int(len(sdf)),
        }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
