#!/usr/bin/env python3
"""Run an STDN diagnostic experiment without modifying frozen vendor files.

This wrapper keeps the existing Backbone pipeline and source lock intact. It
intercepts STDN's ConfigParser reads in-process and optionally forces Adam's
learning rate / weight decay. The normal scripts/run_experiment.py entrypoint
is then executed, so data protocol, adapter, evaluation and output layout stay
identical to the existing Backbone experiments.

Expected location:
    Backbone/STDN_Special/run_diagnostic.py
"""
from __future__ import annotations

import argparse
import configparser
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict


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


def patch_configparser(overrides: Dict[str, Any]) -> None:
    """Override only [Training] keys used by the official STDN recipe.

    SectionProxy.__getitem__ ultimately calls ConfigParser.get(), therefore
    patching get() covers both `section['batch_size']` and parser.get(...).
    """
    original_get = configparser.ConfigParser.get

    mapping = {
        "batch_size": str(overrides["batch_size"]),
        "learning_rate": str(overrides["learning_rate"]),
        "patience": str(overrides["patience"]),
        "epochs": str(overrides["epochs"]),
    }

    def patched_get(self, section, option, *args, **kwargs):  # type: ignore[no-untyped-def]
        if str(section).lower() == "training" and str(option).lower() in mapping:
            return mapping[str(option).lower()]
        return original_get(self, section, option, *args, **kwargs)

    configparser.ConfigParser.get = patched_get  # type: ignore[assignment]


def patch_adam(overrides: Dict[str, Any]) -> None:
    """Force LR/WD for STDN even if the project optimizer factory hard-codes WD.

    STDN's frozen official recipe uses torch.optim.Adam. No vendor file is
    edited, which preserves SOURCE_LOCK verification.
    """
    import torch

    original_adam = torch.optim.Adam
    forced_lr = float(overrides["learning_rate"])
    forced_wd = float(overrides["weight_decay"])

    class DiagnosticAdam(original_adam):
        def __init__(self, params, lr=1e-3, *args, **kwargs):  # type: ignore[no-untyped-def]
            kwargs["weight_decay"] = forced_wd
            super().__init__(params, lr=forced_lr, *args, **kwargs)

    DiagnosticAdam.__name__ = "Adam"
    DiagnosticAdam.__qualname__ = "Adam"
    torch.optim.Adam = DiagnosticAdam  # type: ignore[assignment]


def run_existing_entrypoint(root: Path, stage: str, seed: int, run_stamp: str) -> None:
    script = root / "scripts" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("stdn_special_existing_run_experiment", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root))
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            str(script),
            "--model", "STDN",
            "--stage", stage,
            "--seed", str(seed),
            "--run-stamp", run_stamp,
        ]
        spec.loader.exec_module(module)
        if not hasattr(module, "main"):
            raise RuntimeError("scripts/run_experiment.py has no main()")
        module.main()
    finally:
        sys.argv = old_argv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--stage", default="screening", choices=["screening", "formal"])
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--run-stamp", default=None)
    args = parser.parse_args()

    matrix = load_matrix()
    if args.experiment not in matrix:
        raise SystemExit(f"Unknown experiment {args.experiment!r}. Choices: {', '.join(matrix)}")
    cfg = matrix[args.experiment]
    root = backbone_root()

    # Keep formal runs an explicit choice. Diagnostics should normally remain screening-only.
    if args.stage == "formal" and os.environ.get("STDN_SPECIAL_ALLOW_FORMAL") != "1":
        raise SystemExit("Refusing formal diagnostic by default. Set STDN_SPECIAL_ALLOW_FORMAL=1 explicitly.")

    stamp = args.run_stamp or time.strftime(f"stdn_diag_{args.experiment}_%Y%m%d_%H%M%S")
    print("=" * 88)
    print("STDN_SPECIAL diagnostic")
    print("experiment:", args.experiment)
    print("description:", cfg.get("description", ""))
    print("stage/seed:", args.stage, args.seed)
    print("run_stamp:", stamp)
    print("overrides:", json.dumps(cfg, indent=2))
    print("NOTE: frozen vendor files are not modified; existing Backbone runner is reused.")
    print("=" * 88, flush=True)

    patch_configparser(cfg)
    patch_adam(cfg)
    run_existing_entrypoint(root, args.stage, args.seed, stamp)


if __name__ == "__main__":
    main()
