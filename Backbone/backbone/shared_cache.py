from __future__ import annotations
from pathlib import Path
import sys
from .utils import load_yaml, repository_root


def common_root() -> Path:
    root = repository_root()
    p = root / 'Base_SupportBench'
    if not p.exists():
        raise FileNotFoundError(f'Base_SupportBench not found: {p}')
    for q in (p,):
        if q.exists() and str(q) not in sys.path:
            sys.path.insert(0, str(q))
    return p


def _severity_tag(severity: int | float) -> str:
    value = float(severity)
    if value > 1: value /= 100.0
    return f's{int(round(value * 100)):02d}'


def contract_path(cfg: dict, stage: str, severity: int | float | None = None) -> Path:
    if stage in ('development', 'formal'):
        raw = cfg['shared_contract']['configs'].get(stage) or cfg['shared_contract']['configs'].get('training')
        return Path(raw)
    if stage != 'evaluation':
        raise ValueError(stage)
    if severity is None:
        raise ValueError('evaluation requires severity')
    tag = _severity_tag(severity)
    try:
        return Path(cfg['shared_contract']['configs']['evaluation'][tag])
    except KeyError as e:
        raise ValueError(f'Unsupported evaluation severity {severity}; expected 20/40/60/90') from e


def shared_contract_config(cfg: dict, stage: str, severity: int | float | None = None) -> dict:
    common_root()
    p = contract_path(cfg, stage, severity)
    if not p.exists():
        raise FileNotFoundError(p)
    return load_yaml(p)


def cache_path(cfg: dict, stage: str, severity: int | float | None = None) -> Path:
    common_root()
    from experiment_common.cache import cache_dir
    scfg = shared_contract_config(cfg, stage, severity)
    return cache_dir(scfg, stage)


def validate_shared_cache(cfg: dict, stage: str, severity: int | float | None = None, *, allow_locked_evaluation: bool = False) -> dict:
    """Validate a shared cache with old/new Base_SupportBench cache layer compatibility.

    Training stages intentionally use only the stable two-argument
    ``validate_cache(cfg, stage)`` API.  Older Base_SupportBench cache layer releases on GADI do
    not accept ``allow_locked_evaluation_validation`` and must not receive it.

    Evaluation is intentionally non-gating in the current Backbone workflow.  If
    Base_SupportBench cache layer exposes the newer bypass keyword we use it.  With an older
    Base_SupportBench cache layer, we require the already-materialized cache directory plus its
    READY/manifest files and then normalize its identity without invoking the legacy
    Protocol access gate.
    """
    common_root()
    from experiment_common.cache import validate_cache, cache_dir
    scfg = shared_contract_config(cfg, stage, severity)

    if stage in ('development', 'formal'):
        # Stable API shared by both legacy and current Base_SupportBench cache layer.
        validate_cache(scfg, stage)
    elif stage == 'evaluation':
        import inspect
        params = inspect.signature(validate_cache).parameters
        if 'allow_locked_evaluation_validation' in params:
            validate_cache(
                scfg, stage,
                allow_locked_evaluation_validation=True,
            )
        else:
            # Legacy Base_SupportBench cache layer couples validate_cache() to the Protocol
            # evaluation-access gate.  Backbone simple evaluation no longer uses
            # that gate, so validate the materialized cache presence here and leave
            # detailed tensor-shape checks to the evaluation runner.
            out = cache_dir(scfg, stage)
            if not out.exists():
                raise FileNotFoundError(f'Evaluation cache not found: {out}')
            for name in ('cache_manifest.json', 'READY.json'):
                if not (out / name).exists():
                    raise RuntimeError(f'Missing {name} in evaluation cache: {out}')
    else:
        raise ValueError(stage)

    from .identity import load_cache_identity
    return load_cache_identity(cache_dir(scfg, stage))


def require_evaluation_access(cfg: dict, severity: int | float) -> None:
    """Backward-compatible no-op. Protocol grant files are no longer required."""
    # Keep this symbol so older callers do not break, but deliberately do not call
    # Base_SupportBench cache layer.require_evaluation_cache_access(). Frozen/hash validation is
    # still enforced by validate_shared_cache().
    return None
