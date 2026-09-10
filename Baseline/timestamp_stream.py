#!/usr/bin/env python3
import sys
from datetime import datetime
for line in sys.stdin:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {line.rstrip()}", flush=True)
