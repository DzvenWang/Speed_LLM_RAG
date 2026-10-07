from __future__ import annotations

import os
import re
from pathlib import Path
import yaml

from .paths import project_root

_ENV = re.compile(r"\$\{([^}]+)\}")


def _expand(v):
    if isinstance(v, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(0)), v)
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x) for x in v]
    return v


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else project_root() / "configs" / "fare.yaml"
    with p.open("r", encoding="utf-8") as f:
        return _expand(yaml.safe_load(f))
