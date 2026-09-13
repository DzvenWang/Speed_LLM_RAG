#!/usr/bin/env python3
"""Safely remove obsolete Baseline_exps/_cache symlink trees.

Only symlinks and directories that become empty are removed. Any regular file or
real non-empty directory aborts the cleanup so experiment data cannot be deleted
by mistake.
"""
from pathlib import Path
import sys

ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline')
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.utils import load_yaml

cfg = load_yaml(ROOT / 'configs/baseline.yaml')
legacy = Path(cfg['project']['exp_root']) / '_cache'
if not legacy.exists() and not legacy.is_symlink():
    print('NO LEGACY CACHE ALIAS:', legacy)
    raise SystemExit(0)

for p in sorted(legacy.rglob('*'), key=lambda x: len(x.parts), reverse=True):
    if p.is_symlink():
        print('UNLINK', p, '->', p.resolve(strict=False))
        p.unlink()
remaining_files = [p for p in legacy.rglob('*') if p.is_file() and not p.is_symlink()]
if remaining_files:
    raise SystemExit('REFUSE: real files exist under legacy cache tree: ' + ', '.join(map(str, remaining_files[:20])))
for p in sorted([x for x in legacy.rglob('*') if x.is_dir() and not x.is_symlink()], key=lambda x: len(x.parts), reverse=True):
    try:
        p.rmdir(); print('RMDIR', p)
    except OSError:
        pass
try:
    legacy.rmdir(); print('RMDIR', legacy)
except OSError:
    leftovers = list(legacy.rglob('*'))
    raise SystemExit('REFUSE: non-empty real content remains under legacy cache tree: ' + ', '.join(map(str, leftovers[:20])))
print('LEGACY BASELINE CACHE ALIAS CLEANUP PASS')
