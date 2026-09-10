from __future__ import annotations
import json, os, random, subprocess
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import yaml


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    p=Path(path); p.mkdir(parents=True, exist_ok=True); return p


def log(msg):
    # PBS uses timestamp_stream.py; manual runs still get timestamps.
    if os.environ.get('BACKBONE_TIMESTAMP_STREAM') == '1':
        print(msg, flush=True)
    else:
        print(f"[{datetime.now().strftime('%F %T')}] {msg}", flush=True)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path, obj):
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p,'w',encoding='utf-8') as f: json.dump(obj,f,indent=2,sort_keys=True)


def _ignorable_python_cache_status(line: str) -> bool:
    """Return True only for untracked/generated Python bytecode artifacts.

    Official repositories remain strict for tracked modifications and arbitrary
    untracked files.  We ignore __pycache__, *.pyc and *.pyo because importing
    an untouched official module can create these files on Gadi.
    """
    if not line:
        return True
    path = line[3:].strip() if len(line) >= 4 else line.strip()
    # git may quote unusual paths; these repositories use ordinary paths, but
    # stripping quotes keeps the check harmlessly robust.
    path = path.strip('"')
    parts = Path(path).parts
    return ('__pycache__' in parts) or path.endswith('.pyc') or path.endswith('.pyo')


def git_info(repo):
    repo=Path(repo)
    def run(*args):
        return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()

    raw_status = run('status','--porcelain=v1','--untracked-files=all')
    status_lines = [ln for ln in raw_status.splitlines() if ln.strip()]
    meaningful_dirty = [ln for ln in status_lines if not _ignorable_python_cache_status(ln)]
    ignored_cache = [ln for ln in status_lines if _ignorable_python_cache_status(ln)]
    return {
        'commit': run('rev-parse','HEAD'),
        'branch': run('rev-parse','--abbrev-ref','HEAD'),
        'remote': run('remote','get-url','origin'),
        'dirty': bool(meaningful_dirty),
        'dirty_paths': meaningful_dirty,
        'ignored_python_cache': ignored_cache,
    }


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
