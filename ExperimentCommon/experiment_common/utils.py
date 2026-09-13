from __future__ import annotations
from pathlib import Path
import hashlib, json, yaml


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def write_json(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(',', ':'), default=str).encode())


def manifest_identity(df) -> str:
    cols = [c for c in ['protocol_view_id','view_id','view_group_id','window_start','family','severity','distribution'] if c in df.columns]
    rec = df[cols].copy()
    for c in rec.columns:
        rec[c] = rec[c].astype(str)
    return canonical_sha256(rec.to_dict(orient='records'))
