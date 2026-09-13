from __future__ import annotations
from pathlib import Path
import json,sys
COMMON=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/ExperimentCommon')
if not COMMON.exists(): COMMON=Path(__file__).resolve().parents[2]/'ExperimentCommon'
if str(COMMON) not in sys.path: sys.path.insert(0,str(COMMON))
from experiment_common.cache import load_cache_identity as load_shared_cache_identity,identity_key as shared_identity_key
from experiment_common.utils import canonical_sha256
from .utils import write_json
IDENTITY_SCHEMA_VERSION=4


def load_cache_identity(cache_dir: str|Path, project_version: str|None=None) -> dict:
    s=load_shared_cache_identity(cache_dir)
    return {
      'identity_schema_version':IDENTITY_SCHEMA_VERSION,'project':'Baseline',
      'stage':s['stage'],'dataset':s['dataset'],'protocol_stage_hash':s['protocol_stage_hash'],
      'split_view_hashes':s['split_view_hashes'],'cache_fingerprint':s['cache_fingerprint'],
      'shared_contract_version':s['contract_version'],'data_contract_revision':s['data_contract_revision'],
      'time_feature_revision':s['time_feature_revision'],'metrics_revision':s['metrics_revision'],
      'prediction_contract_revision':s['prediction_contract_revision'],'time_feature_fingerprint':s['time_feature_fingerprint'],
      'time_feature_source':s['time_feature_source'],'time_feature_parser':s['time_feature_parser'],
      'contract_fingerprint':s['contract_fingerprint'],
    }


def identity_key(x:dict)->tuple:
    return (x.get('identity_schema_version'),x.get('stage'),x.get('dataset'),x.get('protocol_stage_hash'),json.dumps(x.get('split_view_hashes',{}),sort_keys=True),x.get('cache_fingerprint'),x.get('shared_contract_version'),x.get('data_contract_revision'),x.get('time_feature_revision'),x.get('metrics_revision'),x.get('prediction_contract_revision'),x.get('time_feature_fingerprint'),x.get('time_feature_source'),x.get('time_feature_parser'),x.get('contract_fingerprint'))


def build_run_manifest(cache_dir,project_version,method,seed,track,extra=None):
    ident=load_cache_identity(cache_dir,project_version); obj={**ident,'code_project_version':None if project_version is None else str(project_version),'method':str(method),'seed':None if seed is None else int(seed),'track':str(track),'status':'PREPARED'}
    if extra: obj.update(extra)
    obj['run_identity_hash']=canonical_sha256({k:v for k,v in obj.items() if k!='run_identity_hash'})
    return obj


def write_run_manifest(path,obj): write_json(path,obj); return Path(path)

def require_same_identity(obj:dict,expected:dict,label='result'):
    if identity_key(obj)!=identity_key(expected): raise RuntimeError(f'{label}: shared Protocol/cache identity mismatch')
