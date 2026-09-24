#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, json
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._common import setup, run_dir, default_tag
from srd.contract import assert_cache_ready
from srd.backbone import load_patchstg_adapter
from srd.data import CacheView
from srd.retrieval import retrieve_queries
from srd.utils import deterministic_ids, write_json, sha256_file


def _mean_top1(r, key):
    return float(np.nanmean(r[key][:, 0])) if key in r else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", required=True)
    ap.add_argument("--stage", default="development", choices=["development", "formal"])
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--queries", type=int, default=None)
    a = ap.parse_args()
    cfg = setup(a.stage)
    assert_cache_ready(cfg, a.stage)
    tag = a.run_tag or default_tag("retrieval")
    out = run_dir(cfg, a.stage, tag) / "retrieval"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adapter, cp, ck, source = load_patchstg_adapter(cfg, a.stage, device, a.checkpoint)
    train = CacheView(cfg, a.stage, "train")
    val = CacheView(cfg, a.stage, "val")
    qids = deterministic_ids(len(val), a.queries or cfg["experiment"]["diagnostic_queries"], cfg["experiment"]["seed"])
    r = retrieve_queries(cfg, adapter, ck, train, val, a.memory, qids, device)
    p = out / "retrieval_topk.npz"
    np.savez_compressed(p, **r)
    summary = {
        "queries": len(qids),
        "top_k": int(r["candidate_ids"].shape[1]),
        "memory": str(Path(a.memory).resolve()),
        "retrieval_file": str(p),
        "sha256": sha256_file(p),
        "checkpoint": str(cp),
        "checkpoint_source": source,
        "mean_top1_observed_mae": _mean_top1(r, "observed_mae"),
        "mean_top1_observed_mae_recent3": _mean_top1(r, "observed_mae_recent3"),
        "mean_top1_observed_mae_recent6": _mean_top1(r, "observed_mae_recent6"),
        "mean_top1_observed_mae_all": _mean_top1(r, "observed_mae_all"),
        "mean_top1_latent_similarity": _mean_top1(r, "latent_sim"),
        "mean_top1_overlap_fraction": _mean_top1(r, "overlap_fraction"),
        "mean_top1_overlap_recent3": _mean_top1(r, "overlap_recent3"),
        "mean_top1_overlap_recent6": _mean_top1(r, "overlap_recent6"),
        "observed_scope_weights": cfg["retrieval"].get("observed_scope_weights", {}),
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
