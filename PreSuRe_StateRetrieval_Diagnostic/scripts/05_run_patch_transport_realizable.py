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
from srd.patch_transport import candidate_proposal, inject_patch, suffix_mask, weighted_topk_proposal
from srd.utils import write_json


def _comp_mae(x, truth, mask):
    m = np.asarray(mask, bool) & np.isfinite(x) & np.isfinite(truth)
    if not np.any(m): return float("nan")
    return float(np.abs(np.asarray(x, np.float32)[m] - np.asarray(truth, np.float32)[m]).mean())


def _forecast(adapter, ck, q, variants, device, bs):
    X = np.stack(variants).astype(np.float32)
    tod = np.repeat(q["tod_hist"][None, :], len(X), axis=0)
    dow = np.repeat(q["dow_hist"][None, :], len(X), axis=0)
    pred = predict_raw_batched(adapter, ck, X, tod, dow, device, batch_size=bs)
    y = np.repeat(q["y_raw"][None, :, :], len(X), axis=0)
    fv = np.repeat(q["future_valid"][None, :, :], len(X), axis=0)
    dm = np.repeat(q["direct_mask"][None, :], len(X), axis=0)
    return masked_mae(pred, y, fv, dm)


def _compose_rank1(q, train, layout, cand_ids, eligible, qpos, L, method, cfg):
    x = q["x_obs"].copy(); used = 0
    for pid, nodes in enumerate(layout.groups):
        if not eligible[qpos, pid]: continue
        cid = int(cand_ids[qpos, pid, 0])
        if cid < 0: continue
        p, f = candidate_proposal(q, train.row(cid), nodes, L, method, cfg)
        if np.any(f):
            x = inject_patch(x, p, f); used += 1
    return x, used


def _compose_weighted(q, train, layout, cand_ids, scores, eligible, qpos, L, K, cfg):
    x = q["x_obs"].copy(); used = 0
    for pid, nodes in enumerate(layout.groups):
        if not eligible[qpos, pid]: continue
        ids = [int(v) for v in cand_ids[qpos, pid, :K] if int(v) >= 0]
        if not ids: continue
        cs = [train.row(cid) for cid in ids]
        ds = scores[qpos, pid, :len(ids)]
        p, f = weighted_topk_proposal(q, cs, ds, nodes, L, cfg, cfg["patch_transport"]["weighted_base_method"])
        if np.any(f):
            x = inject_patch(x, p, f); used += 1
    return x, used


def _compose_hidden_completion_oracle(q, train, layout, cand_ids, eligible, qpos, L, K, method, cfg):
    # Diagnostic ceiling only. Candidate selection sees hidden history truth, never future forecast targets.
    scope = suffix_mask(q["restorable"].shape, L)
    x = q["x_obs"].copy(); used = 0
    for pid, nodes in enumerate(layout.groups):
        if not eligible[qpos, pid]: continue
        target = np.zeros_like(q["restorable"], bool); target[:, nodes] = True; target &= q["restorable"] & scope
        if not np.any(target): continue
        base_err = _comp_mae(q["x_obs"], q["x_full"], target)
        best_err, best = base_err, None
        for r in range(min(K, cand_ids.shape[-1])):
            cid = int(cand_ids[qpos, pid, r])
            if cid < 0: continue
            p, f = candidate_proposal(q, train.row(cid), nodes, L, method, cfg)
            trial = inject_patch(q["x_obs"], p, f)
            err = _comp_mae(trial, q["x_full"], target)
            if np.isfinite(err) and err < best_err:
                best_err, best = err, (p, f)
        if best is not None:
            x = inject_patch(x, best[0], best[1]); used += 1
    return x, used


def main():
    ap = argparse.ArgumentParser(description="Patch transport ablation + realizable Top-1 benchmark")
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--stage", default="development", choices=["development", "formal"])
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--patch-size", type=int, default=None, help="Explicit PatchSTG spatial patch width in reordered slots (debug override)")
    ap.add_argument("--patch-count", type=int, default=None, help="Explicit number of PatchSTG spatial patches (debug override)")
    ap.add_argument("--scopes", default=None)
    ap.add_argument("--methods", default=None)
    ap.add_argument("--weighted-k", default=None)
    ap.add_argument("--oracle-k", type=int, default=20)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    a = ap.parse_args()

    cfg = setup(a.stage); assert_cache_ready(cfg, a.stage)
    scopes = [int(x) for x in (a.scopes.split(",") if a.scopes else cfg["patch_realizable"]["scopes"])]
    methods = [str(x) for x in (a.methods.split(",") if a.methods else cfg["patch_realizable"]["methods"])]
    weighted_k = [int(x) for x in (a.weighted_k.split(",") if a.weighted_k else cfg["patch_realizable"]["weighted_topk"])]
    tag = a.run_tag or default_tag("patchdiag_v060")
    out = run_dir(cfg, a.stage, tag) / "transport_realizable"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adapter, cp, ck, source = load_patchstg_adapter(cfg, a.stage, device, a.checkpoint)
    layout = extract_patch_layout(adapter.model, ck, backbone_root=cfg["project"]["backbone_root"], patch_size_override=a.patch_size, patch_count_override=a.patch_count)
    train, val = CacheView(cfg, a.stage, "train"), CacheView(cfg, a.stage, "val")
    ret = np.load(a.retrieval)
    qids = ret["query_ids"].astype(np.int64)
    cand_ids = ret["candidate_ids"].astype(np.int64)
    scores = ret["score"].astype(np.float32)
    eligible = ret["eligible"].astype(bool)
    oracle_k = max(1, min(int(a.oracle_k), cand_ids.shape[-1]))

    rows = []
    for qpos, qi in enumerate(qids):
        q = val.row(int(qi))
        variants = [q["x_obs"].copy()]
        labels = [("baseline", "none", 0, 0)]
        comp_meta = {}
        for L in scopes:
            target = q["restorable"] & suffix_mask(q["restorable"].shape, L)
            for method in methods:
                x, used = _compose_rank1(q, train, layout, cand_ids, eligible, qpos, L, method, cfg)
                variants.append(x); labels.append(("top1", method, L, 1)); comp_meta[("top1", method, L, 1)] = (_comp_mae(x, q["x_full"], target), used)
                xo, usedo = _compose_hidden_completion_oracle(q, train, layout, cand_ids, eligible, qpos, L, oracle_k, method, cfg)
                variants.append(xo); labels.append(("completion_oracle", method, L, oracle_k)); comp_meta[("completion_oracle", method, L, oracle_k)] = (_comp_mae(xo, q["x_full"], target), usedo)
            for K in weighted_k:
                K = max(1, min(int(K), cand_ids.shape[-1]))
                xw, usedw = _compose_weighted(q, train, layout, cand_ids, scores, eligible, qpos, L, K, cfg)
                name = str(cfg["patch_transport"]["weighted_base_method"])
                variants.append(xw); labels.append(("weighted_topk", name, L, K)); comp_meta[("weighted_topk", name, L, K)] = (_comp_mae(xw, q["x_full"], target), usedw)

        maes = _forecast(adapter, ck, q, variants, device, a.eval_batch_size)
        base_f = float(maes[0])
        for j, label in enumerate(labels[1:], 1):
            mode, method, L, K = label
            cm, used = comp_meta[label]
            base_cm = _comp_mae(q["x_obs"], q["x_full"], q["restorable"] & suffix_mask(q["restorable"].shape, L))
            rows.append({
                "query_id": int(qi), "query_pos": int(qpos), "mode": mode, "method": method,
                "recent_steps": int(L), "K": int(K), "used_patches": int(used),
                "baseline_forecast_mae": base_f, "forecast_mae": float(maes[j]),
                "forecast_utility": base_f - float(maes[j]), "forecast_gain_pct": improvement_pct(base_f, float(maes[j])),
                "baseline_completion_mae": base_cm, "completion_mae": cm,
                "completion_improvement_pct": improvement_pct(base_cm, cm),
            })
        if (qpos + 1) % 10 == 0 or qpos + 1 == len(qids):
            print(f"PATCH TRANSPORT/REALIZABLE {qpos + 1}/{len(qids)}", flush=True)

    df = pd.DataFrame(rows); df.to_csv(out / "per_query.csv", index=False)
    summary_rows = []
    for keys, g in df.groupby(["mode", "method", "recent_steps", "K"], sort=True):
        mode, method, L, K = keys
        b = float(g.baseline_forecast_mae.mean()); m = float(g.forecast_mae.mean())
        bc = float(g.baseline_completion_mae.mean()); cm = float(g.completion_mae.mean())
        summary_rows.append({
            "mode": mode, "method": method, "recent_steps": int(L), "K": int(K),
            "baseline_forecast_mae": b, "forecast_mae": m, "forecast_gain_pct": improvement_pct(b, m),
            "baseline_completion_mae": bc, "completion_mae": cm, "completion_improvement_pct": improvement_pct(bc, cm),
            "beneficial_forecast_query_rate": float((g.forecast_utility > 0).mean()),
            "mean_used_patches": float(g.used_patches.mean()),
        })
    sm = pd.DataFrame(summary_rows); sm.to_csv(out / "summary_table.csv", index=False)
    top1 = sm[sm["mode"] == "top1"]
    payload = {
        "queries": int(len(qids)), "patch_count": int(layout.patch_count), "oracle_k": int(oracle_k),
        "checkpoint": str(cp), "checkpoint_source": source,
        "top1_uses_hidden_truth": False,
        "completion_oracle_uses_hidden_history_truth": True,
        "completion_oracle_uses_future_forecast_target": False,
        "best_top1_gain_pct": float(top1.forecast_gain_pct.max()) if len(top1) else float("nan"),
        "results": summary_rows,
    }
    write_json(out / "summary.json", payload)
    print(sm.to_string(index=False))
    print(json.dumps({k: v for k, v in payload.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
