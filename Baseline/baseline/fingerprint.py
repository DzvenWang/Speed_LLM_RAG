from __future__ import annotations
from pathlib import Path
import hashlib, json
import pandas as pd


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str:
    p=Path(path); h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(',',':'), default=str).encode())


def manifest_identity(df: pd.DataFrame) -> str:
    cols=[c for c in ['view_id','view_group_id','window_start','family','severity','distribution'] if c in df.columns]
    rec=df[cols].copy()
    for c in rec.columns: rec[c]=rec[c].astype(str)
    return canonical_sha256(rec.to_dict(orient='records'))
