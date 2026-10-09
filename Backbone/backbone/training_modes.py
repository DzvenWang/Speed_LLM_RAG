"""Training sensing policies. Does not change shared-cache or evaluation contracts."""
from __future__ import annotations

MODES = ('incomplete', 'complete')


def require_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in MODES:
        raise ValueError(f'Unsupported train_data_mode={value!r}; choose {MODES}')
    return mode


def run_stage_directory(stage: str, mode: str, smoke: bool = False) -> str:
    """Keep all existing incomplete paths unchanged for Baseline consumers."""
    mode = require_mode(mode)
    name = str(stage) if mode == 'incomplete' else f'{stage}_complete'
    return name + ('_smoke' if smoke else '')


def evaluation_mode_subdir(mode: str) -> tuple[str, ...]:
    """Legacy incomplete evaluation locations remain valid; complete is isolated."""
    return ('complete',) if require_mode(mode) == 'complete' else ()


def mode_from_checkpoint(checkpoint: dict) -> str:
    """Historical checkpoints (before dual-mode support) were incomplete."""
    return require_mode(checkpoint.get('train_data_mode', 'incomplete'))
