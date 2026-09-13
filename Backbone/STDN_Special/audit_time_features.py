#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import yaml

from time_features import audit_cache_time_features


def backbone_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    if not (root / "configs" / "backbone_selection.yaml").exists():
        raise SystemExit("STDN_Special must live directly under Backbone/")
    return root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="screening", choices=["screening", "formal", "evaluation"])
    ap.add_argument("--allow-degenerate", action="store_true", help="Report FAIL but return 0. Never use for paper experiments.")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    root = backbone_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from backbone.shared_cache import cache_path
    cfg = yaml.safe_load((root / "configs" / "backbone_selection.yaml").read_text())
    cache = cache_path(cfg, a.stage)
    report = audit_cache_time_features(cache)

    out = Path(a.json_out) if a.json_out else Path(__file__).resolve().parent / "artifacts" / "time_feature_audits" / f"{a.stage}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"TIME_FEATURE_AUDIT={out}")

    if report["status"] != "PASS" and not a.allow_degenerate:
        raise SystemExit(
            "FAIL: Backbone cache time features are degenerate/invalid. Do not run STDN_Special yet. "
            "Rebuild the Backbone cache with scripts/prepare_data.py --stage <stage> --force, then rerun this audit."
        )


if __name__ == "__main__":
    main()
