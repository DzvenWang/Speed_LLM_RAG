#!/usr/bin/env python3
"""Non-destructive cache inventory for STDN diagnostics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def summarize_npz(path: Path):
    import numpy as np
    out = {"path": str(path), "kind": "npz", "arrays": {}}
    with np.load(path, allow_pickle=False) as z:
        for k in z.files:
            a = z[k]
            item = {"shape": list(a.shape), "dtype": str(a.dtype)}
            if a.size and np.issubdtype(a.dtype, np.number):
                finite = np.isfinite(a)
                item.update({
                    "min": float(a[finite].min()) if finite.any() else None,
                    "max": float(a[finite].max()) if finite.any() else None,
                    "mean": float(a[finite].mean()) if finite.any() else None,
                    "zero_fraction": float((a == 0).mean()),
                })
            out["arrays"][k] = item
    return out


def summarize_npy(path: Path):
    import numpy as np
    a = np.load(path, mmap_mode="r", allow_pickle=False)
    item = {"path": str(path), "kind": "npy", "shape": list(a.shape), "dtype": str(a.dtype)}
    if a.size and np.issubdtype(a.dtype, np.number):
        # Sample to keep very large arrays cheap.
        flat = a.reshape(-1)
        s = flat[::max(1, flat.size // 200000)]
        finite = np.isfinite(s)
        item.update({
            "sample_min": float(s[finite].min()) if finite.any() else None,
            "sample_max": float(s[finite].max()) if finite.any() else None,
            "sample_mean": float(s[finite].mean()) if finite.any() else None,
            "sample_zero_fraction": float((s == 0).mean()),
        })
    return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="screening", choices=["screening", "formal"])
    ap.add_argument("--dataset", default="TraffiDent_Net_06")
    a = ap.parse_args()

    backbone = Path(__file__).resolve().parents[1]
    exps = backbone.parent / "Backbone_exps"
    cache = exps / "_cache" / a.dataset / a.stage
    if not cache.exists():
        raise SystemExit(f"Cache not found: {cache}")

    report = {"cache": str(cache), "files": []}
    for p in sorted(cache.rglob("*")):
        if not p.is_file():
            continue
        try:
            if p.suffix == ".npz":
                report["files"].append(summarize_npz(p))
            elif p.suffix == ".npy":
                report["files"].append(summarize_npy(p))
            elif p.suffix in {".json", ".yaml", ".yml", ".csv", ".txt"}:
                report["files"].append({"path": str(p), "kind": p.suffix, "size_bytes": p.stat().st_size})
            else:
                report["files"].append({"path": str(p), "kind": p.suffix or "other", "size_bytes": p.stat().st_size})
        except Exception as exc:
            report["files"].append({"path": str(p), "error": repr(exc)})

    out = Path(__file__).resolve().parent / "artifacts" / f"cache_inventory_{a.stage}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
