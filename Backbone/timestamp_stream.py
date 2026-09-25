#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from datetime import datetime

STAMPED = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]")

for line in sys.stdin:
    # backbone.utils.log() already timestamps its own messages. Preserve those
    # as-is; prefix every other line so the live PBS log has exactly one
    # timestamp on each complete line.
    if STAMPED.match(line):
        out = line
    else:
        out = f"[{datetime.now().strftime('%F %T')}] {line}"
    sys.stdout.write(out)
    sys.stdout.flush()
