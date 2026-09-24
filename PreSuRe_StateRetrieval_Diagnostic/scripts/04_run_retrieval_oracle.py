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
from srd.diagnostics import baseline_mae, candidate_batch
from srd.metrics import improvement_pct
from srd.transport import restoration_scope_name
from srd.utils import write_json, sha256_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--transport", default=None)
    ap.add_argument("--recent-steps", type=int, default=None)
    ap.add_argument("--stage", default="development", choices=["development", "formal"])
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--max-k", type=int, default=None)
    a = ap.parse_args()

    cfg = setup(a.stage)
    if a.recent_steps is not None:
        cfg["transport"]["restoration_scope"] = "recent_suffix"
        cfg["transport"]["recent_steps"] = int(a.recent_steps)
    assert_cache_ready(cfg, a.stage)
    transport = a.transport or cfg["transport"]["default"]
    scope = restoration_scope_name(cfg, cfg["experiment"]["history"])
    tag = a.run_tag or default_tag("retrieval_oracle")
    out = run_dir(cfg, a.stage, tag) / f"oracle_{transport}_{scope}"
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adapter, cp, ck, source = load_patchstg_adapter(cfg, a.stage, device, a.checkpoint)
    train = CacheView(cfg, a.stage, "train")
    val = CacheView(cfg, a.stage, "val")
    r = np.load(a.retrieval)
    qids = r["query_ids"].astype(np.int64)
    cids = r["candidate_ids"].astype(np.int64)
    K = min(cids.shape[1], a.max_k or cids.shape[1])
    k_values = [k for k in map(int, cfg["oracle"]["k_values"]) if k <= K]
    if K not in k_values:
        k_values.append(K)

    base = np.full(len(qids), np.nan, np.float32)
    maes = np.full((len(qids), K), np.nan, np.float32)
    fills = np.zeros((len(qids), K), np.int32)
    for z, qi in enumerate(qids):
        q = val.row(int(qi))
        base[z], _ = baseline_mae(adapter, ck, q, device)
        cand = [train.row(int(j)) for j in cids[z, :K]]
        maes[z], fills[z] = candidate_batch(adapter, ck, q, cand, transport, cfg, device)
        if (z + 1) % 10 == 0 or z + 1 == len(qids):
            print(f"RETRIEVAL ORACLE {z+1}/{len(qids)} K={K} transport={transport} scope={scope}", flush=True)

    evalp = out / "candidate_eval.npz"
    np.savez_compressed(
        evalp,
        query_ids=qids,
        candidate_ids=cids[:, :K],
        baseline_mae=base,
        candidate_mae=maes,
        fill_counts=fills,
        transport=np.asarray(transport),
        restoration_scope=np.asarray(scope),
        recent_steps=np.asarray(int(cfg["transport"].get("recent_steps", 0))),
    )

    b = float(np.nanmean(base))
    summary = {
        "transport": transport,
        "restoration_scope": scope,
        "recent_steps": int(cfg["transport"].get("recent_steps", 0)),
        "queries": len(qids),
        "K": K,
        "checkpoint": str(cp),
        "checkpoint_source": source,
        "baseline_mean_per_query_masked_mae": b,
        "similarity_top1": {},
        "oracle_at_k": {},
        "candidate_only_oracle_at_k": {},
    }

    top1 = float(np.nanmean(maes[:, 0]))
    summary["similarity_top1"] = {
        "mean_per_query_masked_mae": top1,
        "gain_pct": improvement_pct(b, top1),
        "beneficial_query_rate": float(np.nanmean(maes[:, 0] < base)),
        "harmful_query_rate": float(np.nanmean(maes[:, 0] > base)),
        "mean_fill_count": float(np.nanmean(fills[:, 0])),
    }

    for k in sorted(set(k_values)):
        candidate_best = np.nanmin(maes[:, :k], axis=1)
        candidate_mean = float(np.nanmean(candidate_best))
        hit = candidate_best < base
        # Correct oracle: candidate 0 is the untouched PatchSTG baseline.
        safe_best = np.minimum(base, candidate_best)
        safe_mean = float(np.nanmean(safe_best))
        summary["candidate_only_oracle_at_k"][str(k)] = {
            "mean_per_query_masked_mae": candidate_mean,
            "gain_pct": improvement_pct(b, candidate_mean),
            "beneficial_query_rate": float(np.nanmean(hit)),
            "harmful_query_rate": float(np.nanmean(candidate_best > base)),
        }
        summary["oracle_at_k"][str(k)] = {
            "mean_per_query_masked_mae": safe_mean,
            "gain_pct": improvement_pct(b, safe_mean),
            "candidate_hit_rate": float(np.nanmean(hit)),
            "retrieval_used_rate": float(np.nanmean(hit)),
            "no_intervention_rate": float(np.nanmean(~hit)),
            "harmful_query_rate": 0.0,
        }

    summary["candidate_eval_file"] = str(evalp)
    summary["candidate_eval_sha256"] = sha256_file(evalp)
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
