from __future__ import annotations
from .fingerprint import canonical_sha256


def data_config_payload(cfg: dict) -> dict:
    """Only fields that can change materialized tensors belong in cache identity."""
    return {
        'schema': 2,
        'subgraph': cfg['experiment']['subgraph'],
        'history': int(cfg['experiment']['history']),
        'horizon': int(cfg['experiment']['horizon']),
        'severity': float(cfg['experiment']['severity']),
        'families': list(cfg['experiment']['families']),
        'shared_protocol_name': cfg['protocol']['shared_protocol_name'],
        'shared_protocol_version': cfg['protocol']['shared_protocol_version'],
        'consumer_policy': cfg['protocol']['consumer_policy'],
        'no_local_sampling': bool(cfg['protocol']['no_local_sampling']),
        'materializer_version': 1,
    }


def data_config_sha256(cfg: dict) -> str:
    return canonical_sha256(data_config_payload(cfg))
