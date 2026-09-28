from __future__ import annotations
from pathlib import Path
import hashlib,json

BACKBONE_SHARED_KEYS=(
 'dataset','protocol_name','protocol_version','severity','paired_group_id','protocol_stage_hash','split_view_hashes','cache_fingerprint','shared_contract_version','cache_schema_version','data_contract_revision',
 'time_feature_revision','metrics_revision','prediction_contract_revision','time_feature_fingerprint',
 'time_feature_source','time_feature_parser','contract_fingerprint',
)

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def load_frozen_patchstg_registry(path,*,require_files=True,expected_protocol_hash=None,expected_time_feature_revision=None,expected_time_feature_fingerprint=None,expected_time_feature_source=None,expected_dataset='TraffiDent_Net_06',expected_backbone_project_version='1.0.10',expected_backbone_data_contract_revision='traffident_experiment_cache_v1',expected_shared_contract_version='1.2.0',expected_cache_schema_version=2,expected_protocol_version='2.1.0',expected_paired_group_id='TraffiDent_Net_06_core5_multiseverity_v1',expected_metrics_revision='traffident_metrics_v1',expected_prediction_contract_revision='traffident_prediction_v1'):
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(f'Missing Backbone-owned frozen PatchSTG manifest: {p}. Complete Backbone v1.0.10 Core-5 formal reference runs and summarize_formal.py first.')
    obj=json.loads(p.read_text())
    if int(obj.get('schema_version',0))<3: raise RuntimeError('Frozen Backbone manifest must use schema v3+ (Core-5 Backbone).')
    if str(obj.get('project_version'))!=str(expected_backbone_project_version): raise RuntimeError(f'Frozen Backbone project_version={obj.get("project_version")} != {expected_backbone_project_version}')
    if obj.get('selected_model')!='PatchSTG': raise RuntimeError('Frozen backbone must select PatchSTG')
    ident=obj.get('run_identity')
    if not isinstance(ident,dict): raise RuntimeError('Frozen Backbone manifest missing run_identity')
    if str(ident.get('project'))!='Backbone' or str(ident.get('consumer_stage'))!='formal': raise RuntimeError('Frozen PatchSTG must come from Backbone formal stage')
    checks={'dataset':expected_dataset,'data_contract_revision':expected_backbone_data_contract_revision,'shared_contract_version':expected_shared_contract_version,'cache_schema_version':expected_cache_schema_version,'protocol_version':expected_protocol_version,'paired_group_id':expected_paired_group_id,'metrics_revision':expected_metrics_revision,'prediction_contract_revision':expected_prediction_contract_revision}
    if expected_protocol_hash: checks['protocol_stage_hash']=expected_protocol_hash
    if expected_time_feature_revision: checks['time_feature_revision']=expected_time_feature_revision
    if expected_time_feature_fingerprint: checks['time_feature_fingerprint']=expected_time_feature_fingerprint
    if expected_time_feature_source: checks['time_feature_source']=expected_time_feature_source
    for k,v in checks.items():
        if str(ident.get(k))!=str(v): raise RuntimeError(f'Frozen PatchSTG {k}={ident.get(k)} != required {v}')
    cp_raw=str(obj.get('reference_checkpoint','')).strip(); expected_sha=str(obj.get('reference_checkpoint_sha256',''))
    if not cp_raw: raise RuntimeError('Frozen manifest missing reference_checkpoint')
    cp=Path(cp_raw)
    seeds=[int(x) for x in obj.get('formal_seeds',[])]; ref_seed=int(obj.get('reference_seed',-1))
    if ref_seed not in seeds: raise RuntimeError(f'reference_seed={ref_seed} not in formal_seeds={seeds}')
    if not expected_sha: raise RuntimeError('Frozen manifest missing reference_checkpoint_sha256')
    if require_files:
        if not cp.exists(): raise FileNotFoundError(cp)
        got=sha256_file(cp)
        if got!=expected_sha: raise RuntimeError(f'PatchSTG reference checkpoint SHA256 mismatch expected={expected_sha} got={got}')
    return obj

def reference_checkpoint(obj): return Path(obj['reference_checkpoint'])

def require_reference_checkpoint_identity(checkpoint,registry,label='PatchSTG checkpoint'):
    got=checkpoint.get('run_identity'); exp=registry.get('run_identity')
    if not isinstance(got,dict) or not isinstance(exp,dict): raise RuntimeError(f'{label}: missing run_identity')
    for k in BACKBONE_SHARED_KEYS+('consumer_stage','shared_cache_stage'):
        gv=json.dumps(got.get(k,{}),sort_keys=True) if k=='split_view_hashes' else got.get(k)
        ev=json.dumps(exp.get(k,{}),sort_keys=True) if k=='split_view_hashes' else exp.get(k)
        if gv!=ev: raise RuntimeError(f'{label}: run_identity mismatch in {k}')
    if checkpoint.get('model')!='PatchSTG': raise RuntimeError(f'{label}: checkpoint is not PatchSTG')
    if str(checkpoint.get('project_version')) != str(registry.get('project_version')):
        raise RuntimeError(f'{label}: project_version={checkpoint.get("project_version")} != frozen {registry.get("project_version")}')
