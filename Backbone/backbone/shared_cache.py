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


def shared_contract_config(cfg: dict) -> dict:
    """Load the single ExperimentCommon contract used by Backbone."""
    p = Path(cfg['shared_contract']['config'])
    if not p.exists():
        p = COMMON / 'configs' / 'experiment_contract.yaml'
    return load_shared_yaml(p)


def shared_stage(cfg: dict, consumer_stage: str) -> str:
    """Map Backbone's public stage names onto ExperimentCache stage names."""
    try:
        return str(cfg['protocol']['stage_map'][consumer_stage])
    except KeyError as e:
        raise ValueError(f'Unknown Backbone stage: {consumer_stage}') from e


def cache_path(cfg: dict, consumer_stage: str) -> Path:
    """Return the canonical ExperimentCache directory; no Backbone_exps alias exists."""
    scfg = shared_contract_config(cfg)
    return shared_cache_dir(scfg, shared_stage(cfg, consumer_stage))
