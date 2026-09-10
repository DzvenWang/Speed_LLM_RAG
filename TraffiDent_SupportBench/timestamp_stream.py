#!/usr/bin/env python3
from __future__ import annotations

import sys
from datetime import datetime

for raw in sys.stdin:
    line = raw.rstrip("\n")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {line}", flush=True)
