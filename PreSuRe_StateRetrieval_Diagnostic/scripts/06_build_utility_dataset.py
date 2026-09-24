#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, json
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from srd.config import load_config
from srd.utils import write_json


def _v(r, key, i, j, default=np.nan):
    return float(r[key][i, j]) if key in r else float(default)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--candidate-eval", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config(ROOT)
    r = np.load(a.retrieval)
    e = np.load(a.candidate_eval)
    q = r["query_ids"].astype(np.int64)
    c = e["candidate_ids"].astype(np.int64)
    K = c.shape[1]
    if not np.array_equal(q, e["query_ids"]):
        raise RuntimeError("retrieval/eval query IDs differ")

    transport = str(e["transport"].item()) if "transport" in e else "unknown"
    scope = str(e["restoration_scope"].item()) if "restoration_scope" in e else "unknown"
    rows = []
    harm = float(cfg["oracle"]["harmful_margin_mae"])
    for i, qi in enumerate(q):
        for j in range(K):
            base = float(e["baseline_mae"][i])
            model = float(e["candidate_mae"][i, j])
            utility = base - model
            rows.append({
                "query_id": int(qi),
                "candidate_rank": j + 1,
                "candidate_id": int(c[i, j]),
                "transport": transport,
                "restoration_scope": scope,
                "retrieval_score": _v(r, "score", i, j),
                "latent_similarity": _v(r, "latent_sim", i, j),
                "observed_mae": _v(r, "observed_mae", i, j),
                "observed_mae_recent3": _v(r, "observed_mae_recent3", i, j),
                "observed_mae_recent6": _v(r, "observed_mae_recent6", i, j),
                "observed_mae_all": _v(r, "observed_mae_all", i, j),
                "context_score": _v(r, "context_score", i, j),
                "overlap_fraction": _v(r, "overlap_fraction", i, j),
                "overlap_recent3": _v(r, "overlap_recent3", i, j),
                "overlap_recent6": _v(r, "overlap_recent6", i, j),
                "baseline_masked_mae": base,
                "restored_masked_mae": model,
                "utility": utility,
                "beneficial": int(utility > 0),
                "harmful": int(utility < -harm),
                "fill_count": int(e["fill_counts"][i, j]),
            })

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    summary = {
        "transport": transport,
        "restoration_scope": scope,
        "rows": len(df),
        "queries": int(df.query_id.nunique()),
        "candidates_per_query": K,
        "beneficial_rate": float(df.beneficial.mean()),
        "harmful_rate": float(df.harmful.mean()),
        "mean_utility": float(df.utility.mean()),
        "output": str(out.resolve()),
    }
    write_json(out.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
