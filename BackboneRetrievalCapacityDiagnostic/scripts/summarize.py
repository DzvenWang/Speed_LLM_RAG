from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from brcd.metrics import spearman
from brcd.utils import load_yaml, project_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "diagnostic.yaml"))
    ap.add_argument("--target", choices=["speed", "flow"], default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    root = Path(cfg["project"]["exp_root"])
    files = []
    targets = [args.target] if args.target else ["speed", "flow"]
    for target in targets:
        files += list((root / target).glob("**/capacity_results.csv"))
    # Only top-level run summaries, not per-severity duplicates.
    files = [p for p in files if p.parent.name.startswith(("formal_", "smoke_"))]
    if not files:
        raise FileNotFoundError(f"No run-level capacity_results.csv under {root}")
    frames = []
    for p in sorted(files):
        df = pd.read_csv(p); df["run_dir"] = str(p.parent); frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    print(out.to_string(index=False), flush=True)


if __name__ == "__main__": main()
