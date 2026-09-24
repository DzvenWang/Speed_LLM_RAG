#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, json
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._common import setup, run_dir, default_tag
from srd.contract import assert_cache_ready
from srd.backbone import load_patchstg_adapter
from srd.data import CacheView
from srd.evaluate import predict_raw
from srd.metrics import masked_mae, improvement_pct
from srd.utils import deterministic_ids, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="development", choices=["development", "formal"])
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--queries", type=int, default=None)
    a = ap.parse_args()

    cfg = setup(a.stage)
    assert_cache_ready(cfg, a.stage)
    tag = a.run_tag or default_tag("suffix_oracle")
    out = run_dir(cfg, a.stage, tag) / "suffix_oracle"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adapter, cp, ck, source = load_patchstg_adapter(cfg, a.stage, device, a.checkpoint)
    val = CacheView(cfg, a.stage, "val")
    ids = deterministic_ids(len(val), a.queries or cfg["experiment"]["diagnostic_queries"], cfg["experiment"]["seed"])
    H = int(cfg["experiment"]["history"])
    suffixes = sorted(set(max(1, min(H, int(x))) for x in cfg["patch_oracle"]["suffix_lengths"]))

    rows = []
    for z, qi in enumerate(ids, 1):
        q = val.row(qi)
        variants = [q["x_obs"].copy()]
        spans = []
        for L in suffixes:
            s, e = H - L, H
            x = q["x_obs"].copy()
            m = q["restorable"].copy()
            keep = np.zeros_like(m)
            keep[s:e] = True
            m &= keep
            x[m] = q["x_full"][m]
            variants.append(x)
            spans.append((L, s, e, int(m.sum())))

        X = np.stack(variants)
        tod = np.repeat(q["tod_hist"][None, :], len(X), axis=0)
        dow = np.repeat(q["dow_hist"][None, :], len(X), axis=0)
        pred = predict_raw(adapter, ck, X, tod, dow, device)
        y = np.repeat(q["y_raw"][None, :, :], len(X), axis=0)
        fv = np.repeat(q["future_valid"][None, :, :], len(X), axis=0)
        dm = np.repeat(q["direct_mask"][None, :], len(X), axis=0)
        mae = masked_mae(pred, y, fv, dm)
        base = float(mae[0])
        row = {
            "query_id": int(qi),
            "missing_ratio": float(q["restorable"].sum() / max(q["full_valid"].sum(), 1)),
            "baseline_mae": base,
        }
        for j, (L, s, e, fill_count) in enumerate(spans, 1):
            row[f"suffix_{L}_mae"] = float(mae[j])
            row[f"suffix_{L}_utility"] = base - float(mae[j])
            row[f"suffix_{L}_fill_count"] = fill_count
        rows.append(row)
        if z % 25 == 0 or z == len(ids):
            print(f"RECENT-SUFFIX ORACLE {z}/{len(ids)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_query_suffix_oracle.csv", index=False)
    baseline = float(df.baseline_mae.mean())
    summary = {
        "queries": len(df),
        "checkpoint": str(cp),
        "checkpoint_source": source,
        "baseline_mean_per_query_masked_mae": baseline,
        "suffixes": {},
    }
    for L in suffixes:
        model = float(df[f"suffix_{L}_mae"].mean())
        summary["suffixes"][str(L)] = {
            "span": [H - L, H],
            "mean_per_query_masked_mae": model,
            "gain_pct": improvement_pct(baseline, model),
            "beneficial_query_rate": float((df[f"suffix_{L}_utility"] > 0).mean()),
            "mean_fill_count": float(df[f"suffix_{L}_fill_count"].mean()),
        }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
