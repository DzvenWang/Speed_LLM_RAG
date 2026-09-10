#!/usr/bin/env python3
import sys
from datetime import datetime
for line in sys.stdin:
    print(f'[{datetime.now().strftime("%F %T")}] {line}', end='', flush=True)
