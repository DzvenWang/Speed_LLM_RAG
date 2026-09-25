from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


def log(msg: str) -> None:
    # Under PBS, timestamp_stream.py owns the single wall-clock timestamp for
    # every stdout/stderr line. For manual execution, keep timestamped logs.
    if os.environ.get('SUPPORTBENCH_OUTER_TIMESTAMP', '0') == '1':
        print(msg, flush=True)
    else:
        print(f"[{datetime.now().strftime('%F %T')}] {msg}", flush=True)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def stable_seed(*parts: Any, base: int = 20260904) -> int:
    text = '|'.join([str(base), *map(str, parts)])
    digest = hashlib.blake2b(text.encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(digest, 'little') % (2**32 - 1)


def relpath(path: str | Path, root: str | Path) -> str:
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def resolve_rel(root: str | Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (Path(root) / p).resolve()


def link_or_copy(src: str | Path, dst: str | Path, mode: str = 'symlink') -> None:
    src, dst = Path(src), Path(dst)
    if not src.exists():
        return
    ensure_dir(dst.parent)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    mode = mode.lower()
    if mode == 'auto':
        # Windows commonly lacks symlink permission; POSIX/HPC systems benefit
        # from symlinks. If a POSIX symlink still fails, fall back to a copy.
        mode = 'copy' if os.name == 'nt' else 'symlink'
    if mode == 'copy':
        shutil.copy2(src, dst)
    elif mode == 'hardlink':
        os.link(src, dst)
    else:
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def _norm(text: Any) -> str:
    import re
    return re.sub(r'[^a-z0-9]+', '', str(text).lower())


def detect_col(df: pd.DataFrame, aliases: Sequence[str], required: bool = False) -> str | None:
    lookup = {_norm(c): c for c in df.columns}
    for alias in aliases:
        if _norm(alias) in lookup:
            return lookup[_norm(alias)]
    if required:
        raise ValueError(f'Cannot detect one of {aliases}; columns={list(df.columns)}')
    return None


def canonicalize_nodes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out['local_idx'] = np.arange(len(out), dtype=np.int64)
    aliases = {
        'global_node_id': ['global_node_id', 'node_idx', 'node_id', 'station_id', 'sensor_index', 'ID'],
        'fwy': ['Fwy', 'fwy', 'Fwy Name', 'freeway', 'freeway_id', 'route'],
        'direction': ['Direction', 'direction', 'dir'],
        'sensor_type': ['Type', 'type', 'Sensor Type', 'sensor_type'],
        'lat': ['Lat', 'lat', 'latitude'],
        'lng': ['Lng', 'lng', 'lon', 'longitude'],
        'postmile': ['Abs PM', 'Abs_PM', 'postmile', 'Postmile', 'PM', 'milepost', 'Milepost'],
    }
    for target, aliases_ in aliases.items():
        c = detect_col(out, aliases_, required=False)
        if c is not None:
            out[target] = out[c]
    if 'global_node_id' not in out:
        out['global_node_id'] = out['local_idx']
    for c in ['fwy', 'direction', 'sensor_type']:
        if c not in out:
            out[c] = 'unknown'
        out[c] = out[c].astype(str).str.strip()
    for c in ['lat', 'lng', 'postmile']:
        if c not in out:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors='coerce')
    out['road_direction'] = out['fwy'].astype(str) + '__' + out['direction'].astype(str)
    return out


def load_channel_names(src_dir: str | Path, n_channels: int, mapping_file: str = 'channel_mapping.json') -> list[str]:
    mapping = read_json(Path(src_dir) / mapping_file, {}) or {}
    names = mapping.get('output_channel_names')
    if isinstance(names, list) and len(names) == n_channels:
        return [str(x).lower() for x in names]
    defaults = ['flow', 'occupancy', 'speed']
    if n_channels <= len(defaults):
        return defaults[:n_channels]
    return [f'channel_{i}' for i in range(n_channels)]


def uniform_temporal_subsample(ids: np.ndarray, n: int) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    if n <= 0 or len(ids) <= n:
        return ids.copy()
    pos = np.linspace(0, len(ids) - 1, num=n, endpoint=True)
    idx = np.unique(np.rint(pos).astype(np.int64))
    if len(idx) < n:
        missing = np.setdiff1d(np.arange(len(ids)), idx, assume_unique=True)[:n-len(idx)]
        idx = np.sort(np.concatenate([idx, missing]))
    return ids[idx[:n]]


def atomic_save_npy(path: str | Path, arr: np.ndarray) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + '.tmp')
    with tmp.open('wb') as f:
        np.save(f, arr)
    tmp.replace(p)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
