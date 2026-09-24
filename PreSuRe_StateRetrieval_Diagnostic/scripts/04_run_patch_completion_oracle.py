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
from srd.patch_transport import candidate_proposal, inject_patch, suffix_mask
from srd.utils import write_json


def _masked_completion_mae(x, truth, mask):
    m = np.asarray(mask, bool) & np.isfinite(x) & np.isfinite(truth)
    if not np.any(m):
        return float("nan")
    return float(np.abs(np.asarray(x, np.float32)[m] - np.asarray(truth, np.float32)[m]).mean())


def _forecast_maes(adapter, ck, q, variants, device, batch_size):
    X = np.stack(variants).astype(np.float32)
    tod = np.repeat(q["tod_hist"][None, :], len(X), axis=0)
    dow = np.repeat(q["dow_hist"][None, :], len(X), axis=0)
    pred = predict_raw_batched(adapter, ck, X, tod, dow, device, batch_size=batch_size)
    y = np.repeat(q["y_raw"][None, :, :], len(X), axis=0)
    fv = np.repeat(q["future_valid"][None, :, :], len(X), axis=0)
    dm = np.repeat(q["direct_mask"][None, :], len(X), axis=0)
    return masked_mae(pred, y, fv, dm)


def main():
    ap = argparse.ArgumentParser(description="Patch-wise hidden-history completion Oracle@K")
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--stage", default="development", choices=["development", "formal"])
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--patch-size", type=int, default=None, help="Explicit PatchSTG spatial patch width in reordered slots (debug override)")
    ap.add_argument("--patch-count", type=int, default=None, help="Explicit number of PatchSTG spatial patches (debug override)")
    ap.add_argument("--transport", default=None)
    ap.add_argument("--scopes", default=None)
    ap.add_argument("--k-values", default=None)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    a = ap.parse_args()

    cfg = setup(a.stage)
    assert_cache_ready(cfg, a.stage)
    transport = a.transport or cfg["patch_completion_oracle"]["primary_transport"]
    scopes = [int(x) for x in (a.scopes.split(",") if a.scopes else cfg["patch_level"]["suffix_lengths"])]
    k_values = [int(x) for x in (a.k_values.split(",") if a.k_values else cfg["patch_completion_oracle"]["k_values"])]
    allow_abstain = bool(cfg["patch_completion_oracle"].get("allow_abstain_per_patch", True))

    tag = a.run_tag or default_tag("patchdiag_v060")
    out = run_dir(cfg, a.stage, tag) / f"completion_oracle_{transport}"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adapter, cp, ck, source = load_patchstg_adapter(cfg, a.stage, device, a.checkpoint)
    layout = extract_patch_layout(adapter.model, ck, backbone_root=cfg["project"]["backbone_root"], patch_size_override=a.patch_size, patch_count_override=a.patch_count)
    train, val = CacheView(cfg, a.stage, "train"), CacheView(cfg, a.stage, "val")
    ret = np.load(a.retrieval)
    qids = ret["query_ids"].astype(np.int64)
    candidate_ids = ret["candidate_ids"].astype(np.int64)
    eligible = ret["eligible"].astype(bool)
    Kmax = candidate_ids.shape[-1]
    k_values = sorted(set(max(1, min(Kmax, k)) for k in k_values))

    result_rows, selection_rows = [], []
    for qpos, qi in enumerate(qids):
        q = val.row(int(qi))
        variants = [q["x_obs"].copy()]
        variant_keys = [(0, 0)]
        variant_meta = {}
        for L in scopes:
            scope = suffix_mask(q["restorable"].shape, L)
            for K in k_values:
                x_comp = q["x_obs"].copy()
                selected = 0
                candidate_ranks = []
                for pid, nodes in enumerate(layout.groups):
                    target = np.zeros_like(q["restorable"], bool)
                    target[:, nodes] = True
                    target &= q["restorable"] & scope
                    if not np.any(target) or not eligible[qpos, pid]:
                        continue
                    base_patch_err = _masked_completion_mae(q["x_obs"], q["x_full"], target)
                    best_err = base_patch_err if allow_abstain else float("inf")
                    best = None
                    for r in range(min(K, Kmax)):
                        cid = int(candidate_ids[qpos, pid, r])
                        if cid < 0:
                            continue
                        c = train.row(cid)
                        proposal, fill = candidate_proposal(q, c, nodes, L, transport, cfg)
                        trial = inject_patch(q["x_obs"], proposal, fill)
                        err = _masked_completion_mae(trial, q["x_full"], target)
                        if np.isfinite(err) and err < best_err:
                            best_err = err
                            best = (r, cid, proposal, fill)
                    if best is not None:
                        r, cid, proposal, fill = best
                        x_comp = inject_patch(x_comp, proposal, fill)
                        selected += 1
                        candidate_ranks.append(r + 1)
                        selection_rows.append({
                            "query_id": int(qi), "query_pos": int(qpos), "recent_steps": int(L), "K": int(K),
                            "patch_id": int(pid), "selected": True, "candidate_rank": int(r + 1),
                            "candidate_id": int(cid), "baseline_patch_completion_mae": base_patch_err,
                            "selected_patch_completion_mae": float(best_err),
                            "completion_improvement": float(base_patch_err - best_err),
                        })
                    else:
                        selection_rows.append({
                            "query_id": int(qi), "query_pos": int(qpos), "recent_steps": int(L), "K": int(K),
                            "patch_id": int(pid), "selected": False, "candidate_rank": 0, "candidate_id": -1,
                            "baseline_patch_completion_mae": base_patch_err,
                            "selected_patch_completion_mae": base_patch_err,
                            "completion_improvement": 0.0,
                        })
                target_all = q["restorable"] & scope
                base_comp = _masked_completion_mae(q["x_obs"], q["x_full"], target_all)
                comp = _masked_completion_mae(x_comp, q["x_full"], target_all)
                variants.append(x_comp)
                variant_keys.append((L, K))
                variant_meta[(L, K)] = {
                    "base_completion_mae": base_comp,
                    "completion_mae": comp,
                    "selected_patches": selected,
                    "mean_selected_rank": float(np.mean(candidate_ranks)) if candidate_ranks else float("nan"),
                }

        maes = _forecast_maes(adapter, ck, q, variants, device, a.eval_batch_size)
        base_forecast = float(maes[0])
        for j, (L, K) in enumerate(variant_keys[1:], 1):
            m = variant_meta[(L, K)]
            result_rows.append({
                "query_id": int(qi), "query_pos": int(qpos), "recent_steps": int(L), "K": int(K),
                "transport": transport, "baseline_forecast_mae": base_forecast,
                "completed_forecast_mae": float(maes[j]),
                "forecast_utility": base_forecast - float(maes[j]),
                "forecast_gain_pct": improvement_pct(base_forecast, float(maes[j])),
                "baseline_completion_mae": m["base_completion_mae"], "completion_mae": m["completion_mae"],
                "completion_improvement_pct": improvement_pct(m["base_completion_mae"], m["completion_mae"]),
                "selected_patches": int(m["selected_patches"]), "mean_selected_rank": m["mean_selected_rank"],
            })
        if (qpos + 1) % 10 == 0 or qpos + 1 == len(qids):
            print(f"PATCH COMPLETION ORACLE {qpos + 1}/{len(qids)}", flush=True)

    rdf = pd.DataFrame(result_rows)
    sdf = pd.DataFrame(selection_rows)
    rdf.to_csv(out / "per_query_oracle.csv", index=False)
    sdf.to_csv(out / "patch_selections.csv.gz", index=False, compression="gzip")
    summary_rows = []
    for (L, K), g in rdf.groupby(["recent_steps", "K"], sort=True):
        b = float(g.baseline_forecast_mae.mean())
        m = float(g.completed_forecast_mae.mean())
        bc = float(g.baseline_completion_mae.mean())
        cm = float(g.completion_mae.mean())
        summary_rows.append({
            "recent_steps": int(L), "K": int(K), "transport": transport,
            "baseline_forecast_mae": b, "completed_forecast_mae": m,
            "forecast_gain_pct": improvement_pct(b, m),
            "baseline_completion_mae": bc, "completion_mae": cm,
            "completion_improvement_pct": improvement_pct(bc, cm),
            "mean_selected_patches": float(g.selected_patches.mean()),
            "beneficial_forecast_query_rate": float((g.forecast_utility > 0).mean()),
        })
    sm = pd.DataFrame(summary_rows)
    sm.to_csv(out / "oracle_by_k.csv", index=False)
    payload = {
        "queries": int(len(qids)), "patch_count": int(layout.patch_count), "transport": transport,
        "selection_target": "hidden_history_completion_mae", "uses_future_forecast_target_for_selection": False,
        "uses_hidden_history_truth_for_oracle_selection": True, "allow_abstain_per_patch": allow_abstain,
        "checkpoint": str(cp), "checkpoint_source": source,
        "best_downstream_forecast_gain_pct": float(sm.forecast_gain_pct.max()) if len(sm) else float("nan"),
        "results": summary_rows,
    }
    write_json(out / "summary.json", payload)
    print(sm.to_string(index=False))
    print(json.dumps({k: v for k, v in payload.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
