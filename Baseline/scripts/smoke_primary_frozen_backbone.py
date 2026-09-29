#!/usr/bin/env python3
import subprocess,sys
from pathlib import Path
p=Path(__file__).with_name('smoke_primary_group.py')
raise SystemExit(subprocess.call([sys.executable,str(p),'--group','frozen',*sys.argv[1:]]))
