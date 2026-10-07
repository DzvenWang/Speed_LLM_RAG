from __future__ import annotations

import hashlib
import json
import random
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%F %T')}] {msg}", flush=True)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_safe(obj):
    """Recursively convert NumPy scalars and non-finite floats to strict-JSON values.

    Diagnostic metrics may be undefined in stages where the corresponding labels are
    intentionally unavailable (for example alpha_mae during restorer-only training).
    Those values are represented as null rather than NaN so artifact writing can never
    abort a completed training stage.
    """
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def write_json(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, indent=2, sort_keys=True, allow_nan=False)
    tmp.replace(p)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_dict(obj: dict) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def severity_tag(severity: int | float) -> str:
    x = float(severity)
    if x <= 1.0:
        x *= 100.0
    return f"s{int(round(x)):02d}"


def balanced_indices(meta, max_rows: int | None, families: list[str], seed: int = 10) -> np.ndarray:
    n = len(meta)
    if max_rows is None or max_rows <= 0 or max_rows >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    per = max(1, max_rows // max(1, len(families)))
    chosen: list[int] = []
    leftovers: list[int] = []
    fam_values = meta.family.astype(str).to_numpy()
    for fam in families:
        idx = np.flatnonzero(fam_values == str(fam))
        idx = idx.copy(); rng.shuffle(idx)
        chosen.extend(idx[:per].tolist())
        leftovers.extend(idx[per:].tolist())
    if len(chosen) < max_rows:
        rest = np.asarray(leftovers, dtype=np.int64)
        rng.shuffle(rest)
        chosen.extend(rest[: max_rows - len(chosen)].tolist())
    return np.asarray(chosen[:max_rows], dtype=np.int64)


def safe_corr(a, b) -> float | None:
    x = np.asarray(a, dtype=float); y = np.asarray(b, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3:
        return None
    x = x[good]; y = y[good]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])
