"""Every visible Python log line has an ISO-8601 timestamp; robust to partial writes."""
from __future__ import annotations
from datetime import datetime,timezone
import sys
class TimestampWriter:
    def __init__(self,raw):self.raw=raw;self.at_start=True
    def write(self,text):
        s=str(text)
        for piece in s.splitlines(keepends=True):
            if not piece:continue
            if self.at_start:
                self.raw.write('['+datetime.now().astimezone().isoformat(timespec='seconds')+'] ')
                self.at_start=False
            self.raw.write(piece)
            if piece.endswith(('\n','\r')):self.at_start=True
        return len(s)
    def flush(self):return self.raw.flush()
    def __getattr__(self,k):return getattr(self.raw,k)
def install():
    if not isinstance(sys.stdout,TimestampWriter):sys.stdout=TimestampWriter(sys.stdout)
    if not isinstance(sys.stderr,TimestampWriter):sys.stderr=TimestampWriter(sys.stderr)
def utc_now():return datetime.now(timezone.utc).isoformat(timespec='seconds')
