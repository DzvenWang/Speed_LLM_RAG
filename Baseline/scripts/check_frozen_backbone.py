#!/usr/bin/env python3
from pathlib import Path
import subprocess, sys
GADI_ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline')
ROOT = GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[1]
subprocess.check_call([sys.executable, str(ROOT / 'scripts/validate_frozen_patchstg.py')])
