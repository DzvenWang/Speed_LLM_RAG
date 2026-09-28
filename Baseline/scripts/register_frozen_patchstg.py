#!/usr/bin/env python3
"""Deprecated compatibility entrypoint.

Baseline v1.1.0 never creates or overwrites Backbone_exps/FROZEN_BACKBONE.json.
The Backbone project is the sole owner of model selection/freezing. This command
is kept only so old deployment notes fail safe: it performs read-only validation.
"""
from pathlib import Path
import subprocess, sys

GADI_ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline')
ROOT = GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[1]
print('DEPRECATED: Baseline no longer registers PatchSTG. Performing read-only validation instead.')
subprocess.check_call([sys.executable, str(ROOT / 'scripts/validate_frozen_patchstg.py')])
