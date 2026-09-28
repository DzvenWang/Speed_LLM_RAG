from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

ROOT_ENV = 'SPEED_LLM_RAG_ROOT'
ROOT_TOKEN = '${SPEED_LLM_RAG_ROOT}'


def repository_root() -> Path:
    """Resolve the Speed_LLM_RAG repository root without hard-coded paths.

    Priority:
      1. SPEED_LLM_RAG_ROOT environment variable.
      2. Parent directory of this Base_SupportBench project.
    """
    env = os.environ.get(ROOT_ENV, '').strip()
    if env:
        return Path(os.path.expanduser(os.path.expandvars(env))).resolve()
    return Path(__file__).resolve().parents[2]


def _deep_set(d: dict, keys: list[str], value: Any) -> None:
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


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


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parents[1] / 'configs' / 'base_supportbench.yaml'
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg = _expand(cfg)
    if overrides:
        cfg = copy.deepcopy(cfg)
        for key, value in overrides.items():
            _deep_set(cfg, key.split('.'), value)
    cfg['_config_path'] = str(path.resolve())
    cfg['_repository_root'] = str(repository_root())
    return cfg


def project_paths(cfg: dict) -> dict[str, Path]:
    p = cfg['project']
    return {k: Path(v).resolve() for k, v in p.items()}


def discover_source_root(cfg: dict) -> Path:
    explicit = cfg['source'].get('source_root')
    data_root = Path(cfg['project']['data_root']).resolve()
    candidates = []
    if explicit:
        candidates.append(Path(explicit).resolve())
    candidates += [
        data_root / 'processed' / 'traffident' / 'subgraph_datasets_final',
        data_root / 'traffident' / 'subgraph_datasets_final',
        data_root / 'subgraph_datasets_final',
        data_root / 'raw' / 'traffident' / 'subgraph_datasets_final',
    ]
    expected = cfg['source'].get('expected_subgraphs', [])
    checked = []
    for root in candidates:
        checked.append(str(root))
        if not root.exists():
            continue
        found = sorted(p.name for p in root.glob('TraffiDent_Net_*') if p.is_dir())
        if found and (not expected or any(x in found for x in expected)):
            return root
    raise FileNotFoundError('Could not discover TraffiDent source root. Checked:\n  ' + '\n  '.join(checked))
