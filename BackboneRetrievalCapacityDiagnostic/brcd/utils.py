from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT_ENV = "SPEED_LLM_RAG_ROOT"
ROOT_TOKEN = "${SPEED_LLM_RAG_ROOT}"


def project_root() -> Path:
    env = os.environ.get(ROOT_ENV, "").strip()
    if env:
        return Path(os.path.expanduser(os.path.expandvars(env))).resolve()
    # <root>/BackboneRetrievalCapacityDiagnostic/brcd/utils.py
    return Path(__file__).resolve().parents[2]


def diagnostic_root() -> Path:
    return project_root() / "BackboneRetrievalCapacityDiagnostic"


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        root = str(project_root())
        value = value.replace(ROOT_TOKEN, root).replace("$SPEED_LLM_RAG_ROOT", root)
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return _expand(yaml.safe_load(f))


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=True)


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%F %T')}] {msg}", flush=True)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def install_tee(path: str | Path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fh = open(p, "a", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, fh)
    sys.stderr = Tee(sys.__stderr__, fh)
    return fh
