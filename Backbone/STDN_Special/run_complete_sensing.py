#!/usr/bin/env python3
"""Safe complete-sensing control for STDN, enabled only with explicit batch keys.

Run `probe_batch.py` first. This wrapper refuses to guess which tensor is the
incomplete history and which tensor is the corresponding full/clean history.
It replaces exactly one mapping entry at the STDN adapter boundary and then
uses the unchanged Backbone runner/evaluator.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def root() -> Path:
    r = Path(__file__).resolve().parents[1]
    if not (r / "scripts" / "run_experiment.py").exists():
        raise SystemExit("Place STDN_Special directly under Backbone/")
    return r


def same_shape_dtype(a, b) -> bool:
    try:
        return tuple(a.shape) == tuple(b.shape) and getattr(a, "dtype", None) == getattr(b, "dtype", None)
    except Exception:
        return False


def install_complete_wrapper(input_key: str, full_key: str, audit: Path) -> None:
    mod = importlib.import_module("backbone.adapters")
    candidates = [(n, o) for n, o in vars(mod).items() if isinstance(o, type) and "stdn" in n.lower() and hasattr(o, "forward")]
    if not candidates:
        raise RuntimeError("Could not locate an STDN adapter class")

    for cls_name, cls in candidates:
        original = cls.forward
        first = {"done": False}

        def make_wrapper(orig, name):
            def wrapped(self, batch, *args, **kwargs):  # type: ignore[no-untyped-def]
                if not isinstance(batch, dict):
                    raise RuntimeError(
                        f"Complete-sensing wrapper requires dict-like batch; got {type(batch).__name__}. "
                        "Use probe_batch.py and extend this wrapper explicitly rather than guessing."
                    )
                if input_key not in batch or full_key not in batch:
                    raise KeyError(f"Required keys missing. Have={list(batch.keys())}, input_key={input_key}, full_key={full_key}")
                if not same_shape_dtype(batch[input_key], batch[full_key]):
                    raise RuntimeError(
                        f"Refusing replacement because shape/dtype differ: "
                        f"{input_key}={getattr(batch[input_key], 'shape', None)}/{getattr(batch[input_key], 'dtype', None)} vs "
                        f"{full_key}={getattr(batch[full_key], 'shape', None)}/{getattr(batch[full_key], 'dtype', None)}"
                    )
                b2 = dict(batch)
                b2[input_key] = batch[full_key]
                if not first["done"]:
                    record = {
                        "adapter_class": name,
                        "input_key_replaced": input_key,
                        "full_key_used": full_key,
                        "shape": list(batch[input_key].shape),
                        "dtype": str(batch[input_key].dtype),
                        "safety": "Exact same shape/dtype checked before replacement; all other batch fields untouched."
                    }
                    audit.parent.mkdir(parents=True, exist_ok=True)
                    audit.write_text(json.dumps(record, indent=2))
                    print("COMPLETE_SENSING_AUDIT", json.dumps(record), flush=True)
                    first["done"] = True
                return orig(self, b2, *args, **kwargs)
            return wrapped

        cls.forward = make_wrapper(original, cls_name)  # type: ignore[assignment]


def run(stage: str, seed: int, input_key: str, full_key: str, run_stamp: str) -> None:
    r = root()
    sys.path.insert(0, str(r))
    audit = Path(__file__).resolve().parent / "artifacts" / f"complete_sensing_audit_{run_stamp}.json"
    install_complete_wrapper(input_key, full_key, audit)

    script = r / "scripts" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("stdn_complete_existing_entry", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    old = sys.argv[:]
    try:
        sys.argv = [str(script), "--model", "STDN", "--stage", stage, "--seed", str(seed), "--run-stamp", run_stamp]
        spec.loader.exec_module(module)
        module.main()
    finally:
        sys.argv = old


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-key", required=True, help="Batch key actually consumed as incomplete STDN history; obtain from probe.")
    ap.add_argument("--full-key", required=True, help="Batch key containing matched full/clean history; obtain from probe.")
    ap.add_argument("--stage", default="screening", choices=["screening", "formal"])
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--run-stamp", default=None)
    a = ap.parse_args()
    if a.stage != "screening":
        raise SystemExit("Complete-sensing diagnostic is screening-only by design.")
    stamp = a.run_stamp or time.strftime("stdn_diag_complete_%Y%m%d_%H%M%S")
    run(a.stage, a.seed, a.input_key, a.full_key, stamp)


if __name__ == "__main__":
    main()
