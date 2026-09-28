from __future__ import annotations

from pathlib import Path
import sys

COMMON = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/ExperimentCommon')
if not COMMON.exists():
    COMMON = Path(__file__).resolve().parents[2] / 'ExperimentCommon'
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from experiment_common.cache import cache_dir as shared_cache_dir
from experiment_common.utils import load_yaml as load_shared_yaml

VALID_STAGES = ('development', 'formal', 'evaluation')
EVALUATION_SEVERITIES = (20, 40, 60, 90)


def severity_tag(severity: int | float | str) -> str:
    if isinstance(severity, str):
        s = severity.strip().lower()
        if s.startswith('s'):
            s = s[1:]
        value = int(round(float(s)))
    else:
        value = int(round(float(severity)))
        if 0 < float(severity) <= 1:
            value = int(round(float(severity) * 100))
    if value not in EVALUATION_SEVERITIES:
        raise ValueError(f'Unsupported evaluation severity {severity}; expected one of {EVALUATION_SEVERITIES}')
    return f's{value:02d}'


def _resolve_config_path(raw: str | Path) -> Path:
    p = Path(str(raw))
    if p.exists():
        return p
    # Local/project fallback for archives moved off Gadi.
    candidate = COMMON / 'configs' / p.name
    if candidate.exists():
        return candidate
    return p


def shared_contract_config(cfg: dict, stage: str = 'formal', severity: int | None = None) -> dict:
    """Load the ExperimentCommon contract for one Baseline stage.

    development/formal always use the Core-5 s40 training contract.
    evaluation must bind explicitly to one frozen paired severity contract.
    """
    if stage not in VALID_STAGES:
        raise ValueError(f'Unknown Baseline stage: {stage}')
    configs = cfg['shared_contract'].get('configs', {})
    if stage in ('development', 'formal'):
        raw = configs.get('training')
        if not raw:
            raw = cfg['shared_contract'].get('config')
        if not raw:
            raw = COMMON / 'configs' / 'experiment_contract.yaml'
    else:
        if severity is None:
            raise ValueError('evaluation severity is required (20/40/60/90)')
        tag = severity_tag(severity)
        raw = configs.get('evaluation', {}).get(tag)
        if not raw:
            raw = COMMON / 'configs' / f'experiment_contract_eval_{tag}.yaml'
    p = _resolve_config_path(raw)
    if not p.exists():
        raise FileNotFoundError(p)
    return load_shared_yaml(p)


def cache_path(cfg: dict, stage: str, severity: int | None = None) -> Path:
    """Return canonical ExperimentCache path for the requested Core-5 contract."""
    if stage not in VALID_STAGES:
        raise ValueError(f'Unknown Baseline stage: {stage}')
    scfg = shared_contract_config(cfg, stage=stage, severity=severity)
    return shared_cache_dir(scfg, stage)


def protocol_name(cfg: dict, stage: str, severity: int | None = None) -> str:
    if stage in ('development', 'formal'):
        return str(cfg['protocol']['shared_protocol_name'])
    if severity is None:
        raise ValueError('evaluation severity is required')
    sev = int(severity)
    return str(cfg['protocol']['evaluation_protocol_names'][sev])
