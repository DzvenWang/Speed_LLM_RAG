#!/usr/bin/env python3
"""Prefix every line unless already timestamped; handles stdout and stderr pipes."""
from datetime import datetime,timezone
import sys,re
pattern=re.compile(r'^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\]')
for line in sys.stdin:
    msg=line.rstrip('\r\n')
    if pattern.match(msg):print(msg,flush=True)
    else:print(f'[{datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}] {msg}',flush=True)
