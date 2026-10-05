#!/usr/bin/env python3
import subprocess,sys
from pathlib import Path
root=Path(__file__).parent
extra=sys.argv[1:]
rc1=subprocess.call([sys.executable,str(root/"smoke_primary_group.py"),"--group","independent",*extra])
rc2=subprocess.call([sys.executable,str(root/"smoke_primary_group.py"),"--group","reference",*extra])
raise SystemExit(0 if rc1==0 and rc2==0 else 1)
