#!/usr/bin/env python3
from pathlib import Path
import subprocess, sys
ROOT = Path(__file__).resolve().parents[1]
subprocess.check_call([sys.executable, str(ROOT/'scripts'/'source_lock.py'), 'verify', '--root', str(ROOT), '--sources', str(ROOT/'configs'/'backbone_selection.yaml')], cwd=ROOT)
