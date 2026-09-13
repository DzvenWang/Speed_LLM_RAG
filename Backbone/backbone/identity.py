from __future__ import annotations
from pathlib import Path
import json, sys
COMMON=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/ExperimentCommon')
if not COMMON.exists(): COMMON=Path(__file__).resolve().parents[2]/'ExperimentCommon'
if str(COMMON) not in sys.path: sys.path.insert(0,str(COMMON))
from experiment_common.cache import load_cache_identity as load_shared_cache_identity, require_contract, identity_key as shared_identity_key
from experiment_common.utils import load_yaml

IDENTITY_SCHEMA_VERSION = 3


def _shared_cfg(cfg: dict) -> dict:
    p=Path(cfg['shared_contract']['config'])
    if not p.exists(): p=COMMON/'configs/experiment_contract.yaml'
    return load_yaml(p)


def _shared_stage(consumer_stage: str) -> str:
    return {'screening':'development','formal':'formal','evaluation':'evaluation'}[consumer_stage]


def load_cache_identity(cache_dir: str | Path) -> dict:
    shared=load_shared_cache_identity(cache_dir)
    # Project version is intentionally absent from shared cache identity.
    return {
        'identity_schema_version': IDENTITY_SCHEMA_VERSION,
        'project': 'Backbone',
        'consumer_stage': {'development':'screening','formal':'formal','evaluation':'evaluation'}[shared['stage']],
        'shared_cache_stage': shared['stage'],
        'dataset': shared['dataset'],
        'protocol_stage': shared['stage'],
        'protocol_stage_hash': shared['protocol_stage_hash'],
        'split_view_hashes': shared['split_view_hashes'],
        'cache_fingerprint': shared['cache_fingerprint'],
        'shared_contract_version': shared['contract_version'],
        'data_contract_revision': shared['data_contract_revision'],
        'time_feature_revision': shared['time_feature_revision'],
        'metrics_revision': shared['metrics_revision'],
        'prediction_contract_revision': shared['prediction_contract_revision'],
        'time_feature_fingerprint': shared['time_feature_fingerprint'],
        'time_feature_source': shared['time_feature_source'],
        'time_feature_parser': shared['time_feature_parser'],
        'contract_fingerprint': shared['contract_fingerprint'],
    }


def identity_key(x: dict) -> tuple:
    return (
        x.get('identity_schema_version'), x.get('consumer_stage'), x.get('shared_cache_stage'), x.get('dataset'),
        x.get('protocol_stage_hash'), json.dumps(x.get('split_view_hashes',{}),sort_keys=True), x.get('cache_fingerprint'),
        x.get('shared_contract_version'), x.get('data_contract_revision'), x.get('time_feature_revision'),
        x.get('metrics_revision'), x.get('prediction_contract_revision'), x.get('time_feature_fingerprint'),
        x.get('time_feature_source'), x.get('time_feature_parser'), x.get('contract_fingerprint'),
    )


def require_identity(summary: dict, expected: dict, label: str='run') -> None:
    got=summary.get('run_identity')
    if not isinstance(got,dict): raise RuntimeError(f'{label}: missing run_identity')
    if identity_key(got)!=identity_key(expected):
        raise RuntimeError(f'{label}: run identity mismatch.\nexpected={json.dumps(expected,sort_keys=True)}\ngot={json.dumps(got,sort_keys=True)}')


def require_checkpoint_identity(checkpoint: dict, expected: dict, label: str='checkpoint') -> None:
    got=checkpoint.get('run_identity')
    if not isinstance(got,dict): raise RuntimeError(f'{label}: missing run_identity')
    if identity_key(got)!=identity_key(expected):
        raise RuntimeError(f'{label}: checkpoint identity mismatch')


def require_cache_version_compatibility(cfg: dict, identity: dict, consumer_stage: str) -> None:
    if identity.get('consumer_stage') != consumer_stage:
        raise RuntimeError(f'Cache consumer stage={identity.get("consumer_stage")} != {consumer_stage}')
    req=cfg.get('shared_contract', {})
    checks={
        'shared_contract_version': req.get('required_version'),
        'data_contract_revision': req.get('data_contract_revision'),
        'time_feature_revision': req.get('time_feature_revision'),
        'metrics_revision': req.get('metrics_revision'),
        'prediction_contract_revision': req.get('prediction_contract_revision'),
    }
    for key,val in checks.items():
        if val is not None and str(identity.get(key)) != str(val):
            raise RuntimeError(f'{consumer_stage} shared contract {key}={identity.get(key)} != required {val}')
