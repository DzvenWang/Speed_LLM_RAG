from __future__ import annotations
from pathlib import Path
import json, sys
from .utils import repository_root

IDENTITY_SCHEMA_VERSION = 4


def _common_imports():
    common = repository_root() / 'ExperimentCommon'
    if str(common) not in sys.path: sys.path.insert(0, str(common))
    from experiment_common.cache import load_cache_identity as _load
    return _load


def load_cache_identity(cache_dir: str | Path) -> dict:
    shared = _common_imports()(cache_dir)
    return {
        'identity_schema_version': IDENTITY_SCHEMA_VERSION,
        'project': 'Backbone',
        'consumer_stage': shared['stage'],
        'shared_cache_stage': shared['stage'],
        'dataset': shared['dataset'],
        'protocol_name': shared['protocol_name'],
        'protocol_version': shared['protocol_version'],
        'severity': float(shared['severity']),
        'paired_group_id': shared.get('paired_group_id', ''),
        'protocol_stage_hash': shared['protocol_stage_hash'],
        'split_view_hashes': shared['split_view_hashes'],
        'cache_fingerprint': shared['cache_fingerprint'],
        'shared_contract_version': shared['contract_version'],
        'cache_schema_version': int(shared['cache_schema_version']),
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
        x.get('protocol_name'), x.get('protocol_version'), float(x.get('severity', -1)), x.get('paired_group_id'),
        x.get('protocol_stage_hash'), json.dumps(x.get('split_view_hashes', {}), sort_keys=True),
        x.get('cache_fingerprint'), x.get('shared_contract_version'), x.get('cache_schema_version'),
        x.get('data_contract_revision'), x.get('time_feature_revision'), x.get('metrics_revision'),
        x.get('prediction_contract_revision'), x.get('time_feature_fingerprint'), x.get('time_feature_source'),
        x.get('time_feature_parser'), x.get('contract_fingerprint'),
    )


def require_identity(summary: dict, expected: dict, label: str = 'run') -> None:
    got = summary.get('run_identity')
    if not isinstance(got, dict): raise RuntimeError(f'{label}: missing run_identity')
    if identity_key(got) != identity_key(expected):
        raise RuntimeError(f'{label}: run identity mismatch')


def require_checkpoint_identity(checkpoint: dict, expected: dict, label: str = 'checkpoint') -> None:
    got = checkpoint.get('run_identity')
    if not isinstance(got, dict): raise RuntimeError(f'{label}: missing run_identity')
    if identity_key(got) != identity_key(expected):
        raise RuntimeError(f'{label}: checkpoint identity mismatch')


def require_cache_version_compatibility(cfg: dict, identity: dict, stage: str) -> None:
    if identity.get('consumer_stage') != stage:
        raise RuntimeError(f'Cache stage={identity.get("consumer_stage")} != {stage}')
    req = cfg['shared_contract']
    checks = {
        'shared_contract_version': req['required_version'],
        'cache_schema_version': req['required_cache_schema_version'],
        'data_contract_revision': req['data_contract_revision'],
        'time_feature_revision': req['time_feature_revision'],
        'metrics_revision': req['metrics_revision'],
        'prediction_contract_revision': req['prediction_contract_revision'],
        'protocol_version': req['protocol_version'],
        'paired_group_id': req['paired_group_id'],
    }
    for key, val in checks.items():
        if str(identity.get(key)) != str(val):
            raise RuntimeError(f'{stage} cache {key}={identity.get(key)} != required {val}')


def require_formal_eval_compatibility(formal: dict, evaluation: dict) -> None:
    keys = [
        'dataset','shared_contract_version','cache_schema_version','data_contract_revision',
        'time_feature_revision','metrics_revision','prediction_contract_revision',
        'time_feature_fingerprint','time_feature_source','time_feature_parser','paired_group_id',
    ]
    for key in keys:
        if str(formal.get(key)) != str(evaluation.get(key)):
            raise RuntimeError(f'Formal/evaluation contract mismatch in {key}: {formal.get(key)} != {evaluation.get(key)}')
