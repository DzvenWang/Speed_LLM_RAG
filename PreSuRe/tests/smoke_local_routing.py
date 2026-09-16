#!/usr/bin/env python3
# Backward-compatible v0.3 test entry point. v0.4 local routing is node-aware.
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('smoke_node_routing.py')),run_name='__main__')
