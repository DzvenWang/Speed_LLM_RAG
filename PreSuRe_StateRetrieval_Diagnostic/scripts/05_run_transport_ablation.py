#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, json, subprocess
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._common import setup, run_dir
from srd.transport import restoration_scope_name
from srd.utils import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--stage", default="development", choices=["development", "formal"])
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--max-k", type=int, default=10)
    ap.add_argument("--recent-steps", type=int, default=None)
    a = ap.parse_args()

    cfg = setup(a.stage)
    if a.recent_steps is not None:
        cfg["transport"]["recent_steps"] = int(a.recent_steps)
    scope = restoration_scope_name(cfg, cfg["experiment"]["history"])
    script = ROOT / "scripts" / "04_run_retrieval_oracle.py"
    base = run_dir(cfg, a.stage, a.run_tag)
    rows = []

    for method in cfg["transport"]["methods"]:
        cmd = [
            sys.executable, str(script), "--retrieval", a.retrieval,
            "--transport", str(method), "--stage", a.stage,
            "--run-tag", a.run_tag, "--max-k", str(a.max_k),
            "--recent-steps", str(cfg["transport"].get("recent_steps", 3)),
        ]
        if a.checkpoint:
            cmd += ["--checkpoint", a.checkpoint]
        print("RUN", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        summary_path = base / f"oracle_{method}_{scope}" / "summary.json"
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        max_k_key = str(max(map(int, s["oracle_at_k"].keys())))
        rows.append({
            "transport": method,
            "scope": scope,
            "top1_mae": s["similarity_top1"]["mean_per_query_masked_mae"],
            "top1_gain_pct": s["similarity_top1"]["gain_pct"],
            "top1_beneficial_rate": s["similarity_top1"]["beneficial_query_rate"],
            "top1_harmful_rate": s["similarity_top1"]["harmful_query_rate"],
            "oracle_k": int(max_k_key),
            "oracle_mae": s["oracle_at_k"][max_k_key]["mean_per_query_masked_mae"],
            "oracle_gain_pct": s["oracle_at_k"][max_k_key]["gain_pct"],
            "candidate_hit_rate": s["oracle_at_k"][max_k_key]["candidate_hit_rate"],
            "no_intervention_rate": s["oracle_at_k"][max_k_key]["no_intervention_rate"],
        })

    out = base / "transport_ablation"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "transport_ablation.csv", index=False)
    summary = {
        "restoration_scope": scope,
        "recent_steps": int(cfg["transport"].get("recent_steps", 3)),
        "max_k": int(a.max_k),
        "methods": rows,
        "csv": str((out / "transport_ablation.csv").resolve()),
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print("TRANSPORT ABLATION PASS")


if __name__ == "__main__":
    main()
