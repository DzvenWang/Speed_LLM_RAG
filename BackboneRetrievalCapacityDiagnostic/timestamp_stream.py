from __future__ import annotations
import sys
from datetime import datetime
for line in sys.stdin:
    sys.stdout.write(f"[{datetime.now().strftime('%F %T')}] {line}")
    sys.stdout.flush()
