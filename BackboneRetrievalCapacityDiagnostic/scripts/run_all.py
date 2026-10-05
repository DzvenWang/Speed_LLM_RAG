from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["speed", "flow"])
    ap.add_argument("--severities", nargs="+", type=int, default=None)
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--max-memory", type=int, default=None)
    args = ap.parse_args()
    for target in args.targets:
        cmd = [sys.executable, "-u", str(ROOT / "scripts" / "run_diagnostic.py"), "--target", target]
        if args.severities:
            cmd += ["--severities", *[str(x) for x in args.severities]]
        if args.max_queries is not None: cmd += ["--max-queries", str(args.max_queries)]
        if args.max_memory is not None: cmd += ["--max-memory", str(args.max_memory)]
        print("RUN", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__": main()
