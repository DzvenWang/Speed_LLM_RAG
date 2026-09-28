from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT_ENV = 'SPEED_LLM_RAG_ROOT'
ROOT_TOKEN = '${SPEED_LLM_RAG_ROOT}'

ESSENTIAL_COLUMNS = [
    'protocol_view_id', 'view_id', 'view_group_id', 'split', 'window_start',
    'family', 'severity', 'distribution'
]


def repository_root() -> Path:
    env = os.environ.get(ROOT_ENV, '').strip()
    if env:
        return Path(os.path.expanduser(os.path.expandvars(env))).resolve()
    return Path(__file__).resolve().parents[2]


def _expand(v):
    if isinstance(v, str):
        root = str(repository_root())
        v = v.replace(ROOT_TOKEN, root).replace('$SPEED_LLM_RAG_ROOT', root)
        return os.path.expandvars(os.path.expanduser(v))
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x) for x in v]
    return v


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return _expand(yaml.safe_load(f))


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def read_json(path: str | Path) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def canonical_sha256(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(',', ':'), default=str).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_view_hash(df: pd.DataFrame) -> str:
    cols = [c for c in ESSENTIAL_COLUMNS if c in df.columns]
    if 'protocol_view_id' not in cols:
        cols = [c for c in cols if c != 'protocol_view_id']
    rec = df[cols].copy()
    for c in rec.columns:
        rec[c] = rec[c].astype(str)
    return canonical_sha256(rec.to_dict(orient='records'))


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
