#!/usr/bin/env python3
"""Read-only compatibility entrypoint; Backbone owns FROZEN_BACKBONE.json."""
from pathlib import Path
import subprocess,sys
ROOT=next((p for p in Path(__file__).resolve().parents if p.name=='Baseline'),Path(__file__).resolve().parents[1])
print('Baseline does not register/freeze Backbone checkpoints; performing read-only validation.')
subprocess.check_call([sys.executable,str(ROOT/'scripts/validate_frozen_backbone.py')])
