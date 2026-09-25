#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
for s in ('20','40','60','90'):
    cfg = ROOT / 'configs' / f'protocol_s{s}.yaml'
    subprocess.check_call([sys.executable, str(ROOT/'scripts'/'status.py'), '--config', str(cfg)], cwd=ROOT)
