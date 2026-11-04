#!/usr/bin/env python3
from datetime import datetime,timezone
import sys
for line in sys.stdin:
    print(f'[{datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}] {line.rstrip()}',flush=True)
