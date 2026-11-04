"""Every user-visible Python stdout/stderr line gets a UTC timestamp."""
from __future__ import annotations
from datetime import datetime,timezone
import sys,threading

class TimestampStream:
    def __init__(self,stream):
        self.stream=stream;self.pending='';self.lock=threading.RLock()
    def write(self,data):
        if not isinstance(data,str):data=str(data)
        with self.lock:
            self.pending+=data
            while '\n' in self.pending:
                line,self.pending=self.pending.split('\n',1)
                stamp=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                self.stream.write(f'[{stamp}] {line}\n')
                self.stream.flush()
        return len(data)
    def flush(self):
        with self.lock:
            if self.pending:
                stamp=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                self.stream.write(f'[{stamp}] {self.pending}\n')
                self.pending=''
            self.stream.flush()
    def isatty(self):return self.stream.isatty()
    @property
    def encoding(self):return self.stream.encoding

def install_timestamps():
    if not isinstance(sys.stdout,TimestampStream):sys.stdout=TimestampStream(sys.stdout)
    if not isinstance(sys.stderr,TimestampStream):sys.stderr=TimestampStream(sys.stderr)
