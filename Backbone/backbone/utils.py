from __future__ import annotations
import hashlib, json, os, random, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import yaml

ROOT_ENV = 'SPEED_LLM_RAG_ROOT'
ROOT_TOKEN = '${SPEED_LLM_RAG_ROOT}'


def repository_root() -> Path:
    env = os.environ.get(ROOT_ENV, '').strip()
    if env:
        return Path(os.path.expanduser(os.path.expandvars(env))).resolve()
    return Path(__file__).resolve().parents[2]


def _expand(v):
    if isinstance(v, str):
        root = str(repository_root())
        return os.path.expandvars(os.path.expanduser(v.replace(ROOT_TOKEN, root).replace('$SPEED_LLM_RAG_ROOT', root)))
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x) for x in v]
    return v


def load_yaml(path):
    os.environ.setdefault('TRAFFIDENT_TARGET','speed')
    with open(path, 'r', encoding='utf-8') as f:
        return _expand(yaml.safe_load(f))


def ensure_dir(path):
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p


def log(msg):
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


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data); s.flush()
        return len(data)
    def flush(self):
        for s in self.streams: s.flush()


def install_tee(log_path: str | Path):
    p = Path(log_path); p.parent.mkdir(parents=True, exist_ok=True)
    fh = open(p, 'a', encoding='utf-8', buffering=1)
    sys.stdout = Tee(sys.__stdout__, fh)
    sys.stderr = Tee(sys.__stderr__, fh)
    return fh
