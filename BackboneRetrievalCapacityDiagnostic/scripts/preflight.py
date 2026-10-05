from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brcd.backbone_bridge import MODELS, cache_path, discover_formal_artifacts, discover_test_predictions
from brcd.data import discover_history_mask
from brcd.utils import load_yaml, log, project_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["speed", "flow"], default="speed")
    ap.add_argument("--config", default=str(ROOT / "configs" / "diagnostic.yaml"))
    ap.add_argument("--severities", nargs="+", type=int, default=None)
    ap.add_argument("--models", nargs="+", default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    target = args.target
    severities = args.severities or [int(x) for x in cfg["experiment"]["severities"]]
    models = args.models or list(cfg["experiment"]["backbones"])
    root = project_root()
    log(f"PREFLIGHT version={cfg['version']} target={target} root={root}")
    for p in [root / "Backbone", root / "Base_SupportBench", Path(cfg["project"]["diagnostic_root"])]:
        if not p.exists(): raise FileNotFoundError(p)
        log(f"PASS path {p}")
    fc = cache_path(target, "formal")
    log(f"PASS formal_cache {fc}")
    for split in ("train", "val"):
        mp = discover_history_mask(fc, split)
        log(f"PASS history_mask {split} {mp.name}")
    dataset = cfg["experiment"]["dataset"]
    seed = int(cfg["experiment"]["seed"])
    for sev in severities:
        ec = cache_path(target, "evaluation", sev)
        mp = discover_history_mask(ec, "test")
        log(f"PASS evaluation_cache s{sev} {ec} mask={mp.name}")
    for model in models:
        if model not in MODELS: raise ValueError(f"Unsupported model {model}")
        art = discover_formal_artifacts(target, model, dataset, seed)
        log(f"PASS formal model={model} run={art.formal_dir.name}")
        for sev in severities:
            p = discover_test_predictions(art, dataset, sev)
            log(f"PASS test_predictions model={model} s{sev} {p}")
    log("PREFLIGHT PASS")


if __name__ == "__main__":
    main()
