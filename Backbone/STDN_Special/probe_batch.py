#!/usr/bin/env python3
"""Probe the first STDN training batch through the real Backbone pipeline.

The script wraps the STDN adapter at runtime, records batch keys/tensor stats,
and intentionally stops before the first optimizer update. This is designed to
answer adapter questions (TE, masks, missing/full history candidates) without
changing the benchmark cache or frozen vendor source.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


class ProbeComplete(RuntimeError):
    pass


def stats(v: Any) -> Any:
    try:
        import torch
        if torch.is_tensor(v):
            x = v.detach().cpu()
            out = {"type": "tensor", "shape": list(x.shape), "dtype": str(x.dtype)}
            if x.numel():
                if x.dtype == torch.bool:
                    out.update({"true_fraction": float(x.float().mean())})
                elif x.is_floating_point():
                    finite = torch.isfinite(x)
                    out.update({
                        "finite_fraction": float(finite.float().mean()),
                        "min": float(x[finite].min()) if finite.any() else None,
                        "max": float(x[finite].max()) if finite.any() else None,
                        "mean": float(x[finite].float().mean()) if finite.any() else None,
                        "std": float(x[finite].float().std(unbiased=False)) if finite.any() else None,
                        "zero_fraction": float((x == 0).float().mean()),
                    })
                else:
                    out.update({"min": int(x.min()), "max": int(x.max())})
                    if x.numel() <= 20000:
                        u = torch.unique(x)
                        if u.numel() <= 32:
                            out["unique"] = [int(z) for z in u.tolist()]
            return out
    except Exception:
        pass

    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            out = {"type": "ndarray", "shape": list(v.shape), "dtype": str(v.dtype)}
            if v.size and np.issubdtype(v.dtype, np.number):
                finite = np.isfinite(v)
                out.update({
                    "finite_fraction": float(finite.mean()),
                    "min": float(v[finite].min()) if finite.any() else None,
                    "max": float(v[finite].max()) if finite.any() else None,
                    "mean": float(v[finite].mean()) if finite.any() else None,
                    "zero_fraction": float((v == 0).mean()),
                })
            return out
    except Exception:
        pass

    if isinstance(v, dict):
        return {str(k): stats(val) for k, val in v.items()}
    if hasattr(v, "__dict__") and getattr(v, "__dict__", None):
        return {"type": type(v).__name__, "attributes": {str(k): stats(val) for k, val in vars(v).items()}}
    if isinstance(v, (list, tuple)):
        return {"type": type(v).__name__, "len": len(v), "items": [stats(z) for z in v[:8]]}
    return {"type": type(v).__name__, "repr": repr(v)[:300]}


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def install_adapter_probe(out_path: Path) -> None:
    mod = importlib.import_module("backbone.adapters")
    candidates = []
    for name, obj in vars(mod).items():
        if isinstance(obj, type) and "stdn" in name.lower() and hasattr(obj, "forward"):
            candidates.append((name, obj))
    if not candidates:
        names = [n for n, o in vars(mod).items() if isinstance(o, type)]
        raise RuntimeError("Could not locate STDN adapter class. Classes seen: " + ", ".join(names))

    print("STDN adapter candidates:", [n for n, _ in candidates])
    for name, cls in candidates:
        original = cls.forward

        def make_wrapper(orig, cls_name):
            def wrapped(self, batch, *args, **kwargs):  # type: ignore[no-untyped-def]
                payload = {
                    "adapter_class": cls_name,
                    "batch": stats(batch),
                    "notes": [
                        "Inspect integer tensors for DOW range 0..6 and TOD range 0..287.",
                        "Inspect floating input tensors and mask candidates for missing-value convention.",
                        "If both incomplete and full/clean history tensors exist, a complete-sensing diagnostic can be added safely."
                    ],
                }
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(payload, indent=2))
                print(json.dumps(payload, indent=2), flush=True)
                raise ProbeComplete("First STDN batch captured successfully")
            return wrapped

        cls.forward = make_wrapper(original, name)  # type: ignore[assignment]


def run(stage: str, seed: int) -> None:
    r = root()
    if not (r / "scripts" / "run_experiment.py").exists():
        raise SystemExit("Place STDN_Special directly under Backbone/")
    sys.path.insert(0, str(r))
    out = Path(__file__).resolve().parent / "artifacts" / f"batch_probe_{stage}_seed{seed}.json"
    install_adapter_probe(out)

    script = r / "scripts" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("stdn_special_probe_entry", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    old = sys.argv[:]
    try:
        sys.argv = [str(script), "--model", "STDN", "--stage", stage, "--seed", str(seed), "--run-stamp", "stdn_batch_probe"]
        spec.loader.exec_module(module)
        try:
            module.main()
        except ProbeComplete as exc:
            print(f"PASS: {exc}")
            print(f"Probe written to {out}")
            return
    finally:
        sys.argv = old
    raise RuntimeError("Probe did not intercept an STDN forward call")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="screening", choices=["screening", "formal"])
    ap.add_argument("--seed", type=int, default=10)
    a = ap.parse_args()
    run(a.stage, a.seed)


if __name__ == "__main__":
    main()
