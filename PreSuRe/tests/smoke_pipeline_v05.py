#!/usr/bin/env python3
from pathlib import Path
import runpy
# Backward-compatible entry point: v0.7 supersedes the earlier pipeline smoke.
runpy.run_path(str(Path(__file__).with_name('smoke_pipeline_v07.py')),run_name='__main__')
