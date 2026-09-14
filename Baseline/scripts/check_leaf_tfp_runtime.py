#!/usr/bin/env python3
from __future__ import annotations
import importlib, json, sys
mods=['torch','numpy','pandas','tqdm','loguru']
out={'python':sys.executable,'modules':{}}
for m in mods:
    try:
        x=importlib.import_module(m); out['modules'][m]=getattr(x,'__version__','unknown')
    except Exception as e:
        raise SystemExit(f'LEAF_TFP_RUNTIME_FAIL: {m}: {e!r}\nInstall only the missing lightweight dependency into TFP; do not create a separate LEAF environment.')
print(json.dumps(out,indent=2,sort_keys=True)); print('LEAF TFP RUNTIME PASS')
