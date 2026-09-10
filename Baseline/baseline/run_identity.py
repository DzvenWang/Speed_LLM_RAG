from __future__ import annotations
from pathlib import Path
import json
from .fingerprint import canonical_sha256
from .utils import write_json


def load_cache_identity(cache_dir: str | Path, project_version: str | None = None) -> dict:
    p=Path(cache_dir)/'cache_manifest.json'
    if not p.exists(): raise FileNotFoundError(p)
    m=json.loads(p.read_text())
    required=['project_version','stage','protocol_stage_hash','cache_fingerprint','split_protocol_view_hashes']
    missing=[k for k in required if k not in m]
    if missing: raise RuntimeError(f'Cache missing identity fields: {missing}')
    out={'project':'Baseline','project_version':str(m['project_version']),'stage':str(m['stage']),'dataset':str(m.get('subgraph','unknown')),
         'protocol_stage_hash':str(m['protocol_stage_hash']),'split_view_hashes':{str(k):str(v) for k,v in m['split_protocol_view_hashes'].items()},
         'cache_fingerprint':str(m['cache_fingerprint'])}
    if project_version is not None and out['project_version']!=str(project_version):
        raise RuntimeError(f'Cache version {out["project_version"]} != project version {project_version}')
    return out


def identity_key(x: dict) -> tuple:
    return (x.get('project_version'),x.get('stage'),x.get('dataset'),x.get('protocol_stage_hash'),json.dumps(x.get('split_view_hashes',{}),sort_keys=True),x.get('cache_fingerprint'))


def build_run_manifest(cache_dir, project_version, method, seed, track, extra=None):
    ident=load_cache_identity(cache_dir,project_version)
    obj={**ident,'method':str(method),'seed':None if seed is None else int(seed),'track':str(track),'status':'PREPARED'}
    if extra: obj.update(extra)
    obj['run_identity_hash']=canonical_sha256({k:v for k,v in obj.items() if k!='run_identity_hash'})
    return obj


def write_run_manifest(path, obj):
    write_json(path,obj); return Path(path)


def require_same_identity(obj: dict, expected: dict, label='result'):
    if identity_key(obj)!=identity_key(expected):
        raise RuntimeError(f'{label}: Protocol/cache identity mismatch')
