from __future__ import annotations
from pathlib import Path
import sys
COMMON=(next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd())).parent / 'Base_SupportBench'
if not COMMON.exists(): COMMON=Path(__file__).resolve().parents[2]/'Base_SupportBench'
if str(COMMON) not in sys.path: sys.path.insert(0,str(COMMON))
from experiment_common.time_features import ParsedTimeFeatures,resolve_time_features,assert_resolved_time_features
from experiment_common.utils import load_yaml as load_shared_yaml
from .shared_cache import protocol_name, shared_contract_config


def load_protocol_views(cfg, stage: str, severity: int | None = None):
    from shared_protocol.consumer import load_frozen_stage
    return load_frozen_stage(
        cfg['project']['protocol_artifact_root'],
        cfg['experiment']['subgraph'],
        protocol_name(cfg, stage, severity),
        stage,
    )


def resolve_baseline_time_features(bench, cfg: dict) -> ParsedTimeFeatures:
    sc=shared_contract_config(cfg,'formal'); tc=sc['time_features']
    r=resolve_time_features(bench,allow_index_fallback=bool(tc.get('allow_index_modulo_fallback',False)),index_origin_dow=int(tc.get('index_origin_dow',0)))
    assert_resolved_time_features(r)
    return r
