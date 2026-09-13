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


def shared_contract_config(cfg: dict) -> dict:
    """Load the single ExperimentCommon contract used by Baseline."""
    p = Path(cfg['shared_contract']['config'])
    if not p.exists():
        p = COMMON / 'configs' / 'experiment_contract.yaml'
    return load_shared_yaml(p)


def cache_path(cfg: dict, stage: str) -> Path:
    """Return the canonical ExperimentCache directory; no Baseline_exps alias exists."""
    if stage not in VALID_STAGES:
        raise ValueError(f'Unknown Baseline stage: {stage}')
    return shared_cache_dir(shared_contract_config(cfg), stage)
