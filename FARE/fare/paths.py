from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_ENV = "SPEED_LLM_RAG_ROOT"


def repository_root() -> Path:
    env = os.environ.get(ROOT_ENV, "").strip()
    if env:
        return Path(os.path.expanduser(os.path.expandvars(env))).resolve()
    return Path(__file__).resolve().parents[2]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def install_repo_imports() -> Path:
    root = repository_root()
    for p in (root / "Backbone", root / "Base_SupportBench"):
        if not p.exists():
            raise FileNotFoundError(f"Required project not found: {p}")
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return root
