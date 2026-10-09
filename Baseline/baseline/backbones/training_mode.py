"""Select the sensing mode used to train a *reference Backbone checkpoint*.

This does not change Baseline's incomplete-sensing queries, retrieval memory, or
s20/s40/s60/s90 evaluation protocol.  Historical Backbone checkpoints have no
``train_data_mode`` field and are, by definition, ``incomplete``.
"""
from __future__ import annotations

import os
from pathlib import Path

MODES = ('incomplete', 'complete')
ENV_NAME = 'BACKBONE_TRAIN_MODE'


def require_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    if value not in MODES:
        raise ValueError(f'Invalid {ENV_NAME}={mode!r}; expected incomplete or complete')
    return value


def current_mode(requested: str | None = None) -> str:
    if requested is not None:
        return require_mode(requested)
    return require_mode(os.environ.get(ENV_NAME, 'incomplete'))


def set_mode_from_cli(requested: str | None) -> str:
    mode = current_mode(requested)
    os.environ[ENV_NAME] = mode
    return mode


def checkpoint_mode(ck: dict) -> str:
    return require_mode(ck.get('train_data_mode', 'incomplete'))


def reference_stage(mode: str | None = None) -> str:
    return 'formal_complete' if current_mode(mode) == 'complete' else 'formal'


def method_reference_parent(base: Path, model: str, mode: str | None = None) -> Path:
    """Preserve all existing incomplete paths, add /complete for new experiments."""
    path = Path(base) / str(model)
    return path / 'complete' if current_mode(mode) == 'complete' else path
