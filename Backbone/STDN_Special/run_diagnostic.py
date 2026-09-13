#!/usr/bin/env python3
"""STDN diagnostic runner v4 with exact recipe injection and time-cache guard."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict
import yaml

from runtime_overrides import import_and_patch_backbone, source_audit
from time_features import audit_cache_time_features


def backbone_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    required = [root / "scripts" / "run_experiment.py", root / "backbone", root / "vendor" / "STDN"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("STDN_Special must live directly under Backbone/. Missing: " + ", ".join(missing))
    return root


def load_matrix() -> Dict[str, Dict[str, Any]]:
    p = Path(__file__).resolve().parent / "configs" / "experiments.json"
    return json.loads(p.read_text())


def cache_path(root: Path, stage: str) -> Path:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from backbone.shared_cache import cache_path as shared_cache_path
    cfg = yaml.safe_load((root / "configs" / "backbone_selection.yaml").read_text())
    return shared_cache_path(cfg, stage)


def run_existing_entrypoint(root: Path, stage: str, seed: int, run_stamp: str) -> None:
    script = root / "scripts" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("stdn_special_existing_run_experiment", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(script), "--model", "STDN", "--stage", stage, "--seed", str(seed), "--run-stamp", run_stamp]
        spec.loader.exec_module(module)
        if not hasattr(module, "main"):
            raise RuntimeError("scripts/run_experiment.py has no main()")
        module.main()
    finally:
        sys.argv = old_argv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--stage", default="screening", choices=["screening", "formal"])
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--run-stamp", default=None)
    ap.add_argument("--preflight-only", action="store_true")
    a = ap.parse_args()

    matrix = load_matrix()
    if a.experiment not in matrix:
        raise SystemExit(f"Unknown experiment {a.experiment!r}. Choices: {', '.join(matrix)}")
    requested = matrix[a.experiment]
    root = backbone_root()
    special = Path(__file__).resolve().parent

    if a.stage == "formal" and os.environ.get("STDN_SPECIAL_ALLOW_FORMAL") != "1":
        raise SystemExit("Refusing formal diagnostic by default. Set STDN_SPECIAL_ALLOW_FORMAL=1 explicitly.")

    stamp = a.run_stamp or time.strftime(f"stdn_v4_{a.experiment}_%Y%m%d_%H%M%S")
    audit_dir = special / "artifacts" / "runtime_audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    tf_audit = audit_cache_time_features(cache_path(root, a.stage))
    tf_out = special / "artifacts" / "time_feature_audits" / f"{stamp}.json"
    tf_out.parent.mkdir(parents=True, exist_ok=True)
    tf_out.write_text(json.dumps(tf_audit, indent=2))
    print(f"STDN_SPECIAL_TIME_FEATURE_AUDIT={tf_out}")
    if tf_audit["status"] != "PASS" and os.environ.get("STDN_SPECIAL_ALLOW_BAD_TIME_FEATURES") != "1":
        print(json.dumps(tf_audit, indent=2), flush=True)
        raise SystemExit(
            "FAIL: degenerate Backbone TOD/DOW cache. STDN relies directly on TOD/DOW for temporal embedding/dynamic graph. "
            "Fix and rebuild the shared cache before any further paper-eligible STDN experiment."
        )

    runtime_audit: Dict[str, Any] = {
        "version": "4.0",
        "experiment": a.experiment,
        "stage": a.stage,
        "seed": a.seed,
        "run_stamp": stamp,
        "requested": requested,
        "time_feature_audit": str(tf_out),
        "optimizer": None,
        "patch_report": [],
    }

    print("=" * 96)
    print("STDN_SPECIAL v4 diagnostic")
    print("experiment:", a.experiment)
    print("description:", requested.get("description", ""))
    print("stage/seed:", a.stage, a.seed)
    print("run_stamp:", stamp)
    print("REQUESTED_CONFIG:", json.dumps(requested, sort_keys=True))
    print("NOTE: vendor/STDN remains frozen; cfg.models.STDN is overridden only in process memory.")
    print("=" * 96, flush=True)

    sys.path.insert(0, str(root))
    modules = import_and_patch_backbone(requested, runtime_audit["patch_report"], runtime_audit)
    runtime_audit["imported_modules"] = [m.__name__ for m in modules]

    src_out = audit_dir / f"{stamp}.source_audit.json"
    source_audit(root, src_out)
    runtime_audit["source_audit"] = str(src_out)
    audit_out = audit_dir / f"{stamp}.preflight.json"
    audit_out.write_text(json.dumps(runtime_audit, indent=2, default=str))
    print("STDN_SPECIAL_PATCH_REPORT=" + json.dumps(runtime_audit["patch_report"], sort_keys=True), flush=True)

    if a.preflight_only:
        # Exercise the actual patched YAML loader, which is the source of truth
        # for patience/max_epochs in the host runner.
        import backbone.utils as bu
        cfg_doc = bu.load_yaml(root / "configs" / "backbone_selection.yaml")
        runtime_audit["preflight_recipe"] = cfg_doc["models"]["STDN"]
        audit_out.write_text(json.dumps(runtime_audit, indent=2, default=str))
        print(f"PASS preflight-only. Audit: {audit_out}")
        return

    try:
        run_existing_entrypoint(root, a.stage, a.seed, stamp)
    finally:
        runtime_audit["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        audit_out.write_text(json.dumps(runtime_audit, indent=2, default=str))
        print(f"STDN_SPECIAL_RUNTIME_AUDIT={audit_out}", flush=True)


if __name__ == "__main__":
    main()
