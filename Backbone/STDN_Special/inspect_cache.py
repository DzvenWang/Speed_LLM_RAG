#!/usr/bin/env python3
"""Cache inventory + strict TOD/DOW audit for STDN diagnostics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import yaml

from time_features import audit_cache_time_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="screening", choices=["screening", "formal", "evaluation"])
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from backbone.shared_cache import cache_path
    cfg = yaml.safe_load((root / "configs" / "backbone_selection.yaml").read_text())
    cache = cache_path(cfg, a.stage)
    if not cache.exists():
        raise SystemExit(f"Cache not found: {cache}")
    report = audit_cache_time_features(cache)
    out = Path(__file__).resolve().parent / "artifacts" / f"cache_time_audit_{a.stage}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
