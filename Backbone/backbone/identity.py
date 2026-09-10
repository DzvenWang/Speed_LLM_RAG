from __future__ import annotations
from pathlib import Path
import json


def load_cache_identity(cache_dir: str | Path) -> dict:
    cache = Path(cache_dir)
    p = cache / 'cache_manifest.json'
    if not p.exists():
        raise FileNotFoundError(p)
    m = json.loads(p.read_text())
    required = ['project_version', 'backbone_stage', 'protocol_stage', 'protocol_stage_hash', 'cache_fingerprint', 'split_view_hashes']
    missing = [k for k in required if k not in m]
    if missing:
        raise RuntimeError(f'Cache manifest missing identity fields: {missing}')
    return {
        'project': 'Backbone',
        'project_version': str(m['project_version']),
        'consumer_stage': str(m['backbone_stage']),
        'protocol_stage': str(m['protocol_stage']),
        'protocol_stage_hash': str(m['protocol_stage_hash']),
        'split_view_hashes': {str(k): str(v) for k, v in m['split_view_hashes'].items()},
        'cache_fingerprint': str(m['cache_fingerprint']),
        'dataset': str(m.get('subgraph', 'unknown')),
    }


def identity_key(identity: dict) -> tuple:
    return (
        identity.get('project_version'), identity.get('consumer_stage'),
        identity.get('protocol_stage'), identity.get('protocol_stage_hash'),
        json.dumps(identity.get('split_view_hashes', {}), sort_keys=True),
        identity.get('cache_fingerprint'), identity.get('dataset'),
    )


def require_identity(summary: dict, expected: dict, label: str = 'run') -> None:
    got = summary.get('run_identity')
    if not isinstance(got, dict):
        raise RuntimeError(f'{label}: missing run_identity; old/untracked result is not eligible.')
    if identity_key(got) != identity_key(expected):
        raise RuntimeError(
            f'{label}: run identity mismatch.\nexpected={json.dumps(expected, sort_keys=True)}\n'
            f'got={json.dumps(got, sort_keys=True)}'
        )


def allowed_cache_project_versions(cfg: dict, consumer_stage: str) -> list[str]:
    key = f'{consumer_stage}_cache_project_versions'
    vals = cfg.get('compatibility', {}).get(key, [str(cfg.get('version'))])
    out = []
    for x in vals:
        sx = str(x)
        if sx not in out:
            out.append(sx)
    return out


def require_cache_version_compatibility(cfg: dict, identity: dict, consumer_stage: str) -> None:
    got = str(identity.get('project_version'))
    allowed = allowed_cache_project_versions(cfg, consumer_stage)
    if got not in allowed:
        raise RuntimeError(
            f'{consumer_stage} cache project version {got} is not compatible with code/config '
            f'version {cfg.get("version")}; allowed={allowed}'
        )
