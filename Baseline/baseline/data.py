from __future__ import annotations
from pathlib import Path
import sys
COMMON=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/ExperimentCommon')
if not COMMON.exists(): COMMON=Path(__file__).resolve().parents[2]/'ExperimentCommon'
if str(COMMON) not in sys.path: sys.path.insert(0,str(COMMON))
from experiment_common.time_features import ParsedTimeFeatures,resolve_time_features,assert_resolved_time_features
from experiment_common.utils import load_yaml as load_shared_yaml


def load_protocol_views(cfg, stage: str):
    from shared_protocol.consumer import load_frozen_stage
    return load_frozen_stage(cfg['project']['protocol_artifact_root'],cfg['experiment']['subgraph'],cfg['protocol']['shared_protocol_name'],stage)


def resolve_baseline_time_features(bench, cfg: dict) -> ParsedTimeFeatures:
    cp=Path(cfg['shared_contract']['config'])
    if not cp.exists(): cp=COMMON/'configs/experiment_contract.yaml'
    sc=load_shared_yaml(cp); tc=sc['time_features']
    r=resolve_time_features(bench,allow_index_fallback=bool(tc.get('allow_index_modulo_fallback',False)),index_origin_dow=int(tc.get('index_origin_dow',0)))
    assert_resolved_time_features(r)
    return r
