#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from srd.config import load_config


def latest_v03(cfg, stage):
    base = Path(cfg["project"]["exp_root"]) / cfg["experiment"]["subgraph"] / stage
    if not base.exists():
        return None
    runs = sorted([p for p in base.glob("srd_v03.*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    for p in runs:
        if (p / "retrieval" / "retrieval_topk.npz").exists() and (p / "window_ceiling_sweep" / "summary.json").exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description="SRD v0.4 realizable benchmark preflight")
    ap.add_argument("--stage", default="development", choices=["development", "formal"])
    ap.add_argument("--source-run", default=None)
    ap.add_argument("--max-k", type=int, default=50)
    ap.add_argument("--scopes", default="1,2,3,4,6")
    a = ap.parse_args()
    cfg = load_config(ROOT)
    try:
        import sklearn
        sk = sklearn.__version__
    except Exception as exc:
        raise SystemExit(f"FAIL scikit-learn: {exc}")

    source = Path(a.source_run).resolve() if a.source_run else latest_v03(cfg, a.stage)
    if source is None:
        raise SystemExit(
            "FAIL: no completed srd_v03.* experiment found. Run v0.3 first or pass --source-run <path>."
        )
    scopes = [int(x.strip()) for x in a.scopes.split(",") if x.strip()]
    checks = {
        "source_run": source,
        "retrieval": source / "retrieval" / "retrieval_topk.npz",
        "ceiling_summary": source / "window_ceiling_sweep" / "summary.json",
    }
    for L in scopes:
        csv = source / f"oracle_raw_copy_recent{L}" / "utility_dataset.csv"
        npz = source / f"oracle_raw_copy_recent{L}" / "candidate_eval.npz"
        checks[f"recent{L}_utility_or_eval"] = csv if csv.exists() else npz
    ok = True
    print("PreSuRe State-Retrieval Diagnostic v0.4 preflight")
    print("scikit-learn:", sk)
    print("max_k:", a.max_k)
    print("scopes:", scopes)
    for name, p in checks.items():
        exists = Path(p).exists()
        print(("PASS" if exists else "FAIL"), name, "=", p)
        ok = ok and exists
    if not ok:
        raise SystemExit(2)
    print("SOURCE_RUN=", source)
    print("PREFLIGHT PASS")


if __name__ == "__main__":
    main()
