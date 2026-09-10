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
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p


def log(msg):
    if os.environ.get('BASELINE_TIMESTAMP_STREAM') == '1':
        print(msg, flush=True)
    else:
        print(f"[{datetime.now().strftime('%F %T')}] {msg}", flush=True)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _ignorable_python_cache_status(line: str) -> bool:
    if not line:
        return True
    path = line[3:].strip() if len(line) >= 4 else line.strip()
    path = path.strip('"')
    parts = Path(path).parts
    return ('__pycache__' in parts) or path.endswith('.pyc') or path.endswith('.pyo')


def git_info(repo):
    repo = Path(repo)
    def run(*args):
        return subprocess.check_output(['git','-C',str(repo),*args], text=True).strip()
    raw = run('status','--porcelain=v1','--untracked-files=all')
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    dirty = [ln for ln in lines if not _ignorable_python_cache_status(ln)]
    ignored = [ln for ln in lines if _ignorable_python_cache_status(ln)]
    return {
        'commit': run('rev-parse','HEAD'),
        'branch': run('rev-parse','--abbrev-ref','HEAD'),
        'remote': run('remote','get-url','origin'),
        'dirty': bool(dirty),
        'dirty_paths': dirty,
        'ignored_python_cache': ignored,
    }
