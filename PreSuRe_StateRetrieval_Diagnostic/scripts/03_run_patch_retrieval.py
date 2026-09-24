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
from srd.patch_layout import extract_patch_layout
from srd.patch_retrieval import retrieve_patch
from srd.patch_transport import suffix_mask
from srd.utils import deterministic_ids, write_json


def main():
    ap = argparse.ArgumentParser(description="Mask-conditioned patch-local historical retrieval")
    ap.add_argument("--stage", default="development", choices=["development", "formal"])
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--patch-size", type=int, default=None, help="Explicit PatchSTG spatial patch width in reordered slots (debug override)")
    ap.add_argument("--patch-count", type=int, default=None, help="Explicit number of PatchSTG spatial patches (debug override)")
    ap.add_argument("--queries", type=int, default=None)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    a = ap.parse_args()

    cfg = setup(a.stage)
    assert_cache_ready(cfg, a.stage)
    if a.top_k is not None:
        cfg["patch_retrieval"]["top_k"] = int(a.top_k)
    tag = a.run_tag or default_tag("patchdiag_v060")
    out = run_dir(cfg, a.stage, tag) / "patch_retrieval"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adapter, cp, ck, source = load_patchstg_adapter(cfg, a.stage, device, a.checkpoint)
    layout = extract_patch_layout(adapter.model, ck, backbone_root=cfg["project"]["backbone_root"], patch_size_override=a.patch_size, patch_count_override=a.patch_count)
    train = CacheView(cfg, a.stage, "train")
    val = CacheView(cfg, a.stage, "val")
    qids = deterministic_ids(len(val), a.queries or cfg["experiment"]["diagnostic_queries"], cfg["experiment"]["seed"])
    max_windows = int(a.max_windows or cfg["patch_retrieval"]["max_train_windows"])
    tids = deterministic_ids(len(train), max_windows, cfg["experiment"]["seed"])
    K = min(int(cfg["patch_retrieval"]["top_k"]), len(tids))
    Q, P = len(qids), layout.patch_count

    cand = np.full((Q, P, K), -1, dtype=np.int64)
    score = np.full((Q, P, K), np.inf, dtype=np.float32)
    level = np.full_like(score, np.inf)
    delta = np.full_like(score, np.inf)
    overlap = np.zeros_like(score)
    tod_dist = np.ones_like(score)
    dow_match = np.zeros_like(score)
    eligible = np.zeros((Q, P), dtype=bool)
    restorable_count = np.zeros((Q, P), dtype=np.int32)
    observed_count = np.zeros((Q, P), dtype=np.int32)
    rows = []
    max_scope = max(int(x) for x in cfg["patch_level"]["suffix_lengths"])

    for qpos, qi in enumerate(qids):
        q = val.row(int(qi))
        scope = suffix_mask(q["restorable"].shape, max_scope)
        for pid, nodes in enumerate(layout.groups):
            rcount = int((q["restorable"][:, nodes] & scope[:, nodes]).sum())
            ocount = int(q["history_observed"][:, nodes].sum())
            restorable_count[qpos, pid] = rcount
            observed_count[qpos, pid] = ocount
            if rcount <= 0:
                continue
            res = retrieve_patch(q, train, tids, nodes, cfg)
            k = min(K, len(res.candidate_ids))
            if k <= 0:
                continue
            eligible[qpos, pid] = True
            cand[qpos, pid, :k] = res.candidate_ids[:k]
            score[qpos, pid, :k] = res.score[:k]
            level[qpos, pid, :k] = res.level_mae[:k]
            delta[qpos, pid, :k] = res.delta_mae[:k]
            overlap[qpos, pid, :k] = res.overlap_fraction[:k]
            tod_dist[qpos, pid, :k] = res.tod_distance[:k]
            dow_match[qpos, pid, :k] = res.dow_match[:k]
            for r in range(k):
                rows.append({
                    "query_pos": int(qpos), "query_id": int(qi), "patch_id": int(pid), "rank": int(r + 1),
                    "candidate_id": int(res.candidate_ids[r]), "score": float(res.score[r]),
                    "level_mae": float(res.level_mae[r]), "delta_mae": float(res.delta_mae[r]),
                    "overlap_fraction": float(res.overlap_fraction[r]), "tod_distance": float(res.tod_distance[r]),
                    "dow_match": float(res.dow_match[r]), "patch_restorable_count": rcount,
                    "patch_observed_count": ocount,
                })
        if (qpos + 1) % 10 == 0 or qpos + 1 == Q:
            print(f"PATCH RETRIEVAL {qpos + 1}/{Q}", flush=True)

    np.savez_compressed(
        out / "patch_retrieval_topk.npz",
        query_ids=qids.astype(np.int64),
        train_ids=tids.astype(np.int64),
        candidate_ids=cand,
        score=score,
        level_mae=level,
        delta_mae=delta,
        overlap_fraction=overlap,
        tod_distance=tod_dist,
        dow_match=dow_match,
        eligible=eligible,
        restorable_count=restorable_count,
        observed_count=observed_count,
    )
    pd.DataFrame(rows).to_csv(out / "patch_retrieval_topk.csv.gz", index=False, compression="gzip")
    summary = {
        "queries": int(Q), "patch_count": int(P), "top_k": int(K), "train_memory_windows": int(len(tids)),
        "eligible_query_patch_pairs": int(eligible.sum()),
        "queries_with_any_eligible_patch": int(eligible.any(axis=1).sum()),
        "mean_eligible_patches_per_query": float(eligible.sum(axis=1).mean()),
        "checkpoint": str(cp), "checkpoint_source": source,
        "ranking": {
            "mask_conditioned": True,
            "uses_hidden_query_state": False,
            "level_weight": float(cfg["patch_retrieval"]["level_weight"]),
            "delta_weight": float(cfg["patch_retrieval"]["delta_weight"]),
            "tod_penalty_weight": float(cfg["patch_retrieval"]["tod_penalty_weight"]),
            "dow_penalty_weight": float(cfg["patch_retrieval"]["dow_penalty_weight"]),
        },
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"RETRIEVAL={out / 'patch_retrieval_topk.npz'}")


if __name__ == "__main__":
    main()
