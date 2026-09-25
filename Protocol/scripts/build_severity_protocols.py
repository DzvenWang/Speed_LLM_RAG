#!/usr/bin/env python3
"""Backward-compatible alias for the v2 paired evaluation builder."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
raise SystemExit(subprocess.call([
    sys.executable,
    str(ROOT / 'scripts' / 'build_multiseverity_protocols.py'),
    '--evaluation-only',
    *sys.argv[1:],
], cwd=ROOT))
