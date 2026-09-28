#!/usr/bin/env python3
from pathlib import Path
import subprocess
ROOT=next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd())
subprocess.check_call(['python',str(ROOT/'scripts/source_lock.py'),'verify','--root',str(ROOT),'--sources',str(ROOT/'configs/official_sources.yaml')])
