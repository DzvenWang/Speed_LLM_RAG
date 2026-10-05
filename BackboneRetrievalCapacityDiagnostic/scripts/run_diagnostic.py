from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brcd.diagnostic import run
from brcd.utils import ensure_dir, install_tee, load_yaml, log, now_stamp, project_root, write_json


def main():
    ap = argparse.ArgumentParser(description="Backbone-Retrieval Capacity Diagnostic")
    ap.add_argument("--target", choices=["speed", "flow"], required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "diagnostic.yaml"))
    ap.add_argument("--severities", nargs="+", type=int, default=None)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--run-stamp", default=None)
    ap.add_argument("--max-queries", type=int, default=None, help="Smoke/debug only; bounds val/test queries")
    ap.add_argument("--max-memory", type=int, default=None, help="Smoke/debug only; bounds Train retrieval memory")
    ap.add_argument("--bootstrap-samples", type=int, default=None)
    ap.add_argument("--no-tee", action="store_true", help="PBS wrapper already tees stdout")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    target = args.target
    os.environ["TRAFFIDENT_TARGET"] = target
    seed = int(cfg["experiment"]["seed"])
    severities = args.severities or [int(x) for x in cfg["experiment"]["severities"]]
    models = args.models or list(cfg["experiment"]["backbones"])
    stamp = args.run_stamp or now_stamp()
    mode = "smoke" if args.max_queries is not None or args.max_memory is not None else "formal"
    run_dir = ensure_dir(Path(cfg["project"]["exp_root"]) / target / cfg["experiment"]["dataset"] / f"{mode}_{stamp}_seed{seed}")
    if not args.no_tee:
        log_dir = ensure_dir(Path(cfg["project"]["log_root"]) / target)
        install_tee(log_dir / f"BRCD.{target}.{mode}.seed{seed}.{stamp}.live.log")
    write_json(run_dir / "RUN_MANIFEST.json", {
        "version": cfg["version"], "target": target, "seed": seed, "mode": mode,
        "severities": severities, "models": models, "max_queries": args.max_queries,
        "max_memory": args.max_memory, "status": "STARTED",
    })
    log(f"DIAGNOSTIC_START version={cfg['version']} target={target} mode={mode} seed={seed} out={run_dir}")
    report = run(
        cfg, target=target, severities=severities, models=models, seed=seed, run_dir=run_dir,
        max_queries=args.max_queries, max_memory=args.max_memory, bootstrap_samples=args.bootstrap_samples,
    )
    write_json(run_dir / "RUN_MANIFEST.json", {
        "version": cfg["version"], "target": target, "seed": seed, "mode": mode,
        "severities": severities, "models": models, "max_queries": args.max_queries,
        "max_memory": args.max_memory, "status": "FINISHED", "report": report,
    })
    log(f"DIAGNOSTIC_PASS target={target} out={run_dir}")


if __name__ == "__main__":
    main()
