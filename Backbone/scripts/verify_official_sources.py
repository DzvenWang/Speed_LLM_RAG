#!/usr/bin/env python3
"""Verify the frozen PatchSTG source tree using the current source_lock CLI."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "backbone.yaml"

subprocess.check_call(
    [
        sys.executable,
        str(ROOT / "scripts" / "source_lock.py"),
        "verify",
        "--config",
        str(CONFIG),
    ],
    cwd=ROOT,
)
