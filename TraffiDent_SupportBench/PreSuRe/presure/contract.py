from __future__ import annotations
from pathlib import Path
import json, sys
from .utils import load_yaml

def _common(cfg):
    p=Path(cfg['project']['common_root'])
    if str(p) not in sys.path: sys.path.insert(0,str(p))
    return p

def cache_dir(cfg,stage):
    return Path(cfg['project']['cache_root'])/cfg['experiment']['subgraph']/stage

def load_cache_identity(cfg,stage):
    _common(cfg)
    from experiment_common.cache import load_cache_identity as _load, require_contract
    cp=Path(cfg['shared_contract']['config'])
    cc=load_yaml(cp)
    p=cache_dir(cfg,stage)
    ident=_load(p); require_contract(cc,ident,stage)
    return ident

def assert_cache_ready(cfg,stage):
    p=cache_dir(cfg,stage)
    if not (p/'READY.json').exists(): raise RuntimeError(f'ExperimentCache not ready: {p}')
    ident=load_cache_identity(cfg,stage)
    expected={'development':{'train','val'},'formal':{'train','val'}}.get(stage)
    if expected is None: raise ValueError('PreSuRe v0.2 only supports development/formal; Test is intentionally gated.')
    for split in expected:
        for n in ['x_obs_norm','history_observed','y_raw','future_valid','direct_mask','support_mask','tod_hist','dow_hist']:
            if not (p/f'{split}_{n}.npy').exists(): raise RuntimeError(f'Missing cache array {split}_{n}.npy')
        if not (p/f'{split}_manifest.csv').exists(): raise RuntimeError(f'Missing {split}_manifest.csv')
    return ident

def compare_checkpoint_identity(ck,cache_ident,stage):
    rid=ck.get('run_identity')
    if not isinstance(rid,dict): raise RuntimeError('PatchSTG checkpoint missing run_identity')
    shared_stage=rid.get('shared_cache_stage')
    if str(shared_stage)!=str(stage):
        raise RuntimeError(f'Backbone checkpoint shared_cache_stage={shared_stage} != PreSuRe stage={stage}. Refusing cross-stage leakage.')
    keys=['cache_fingerprint','protocol_stage_hash','shared_contract_version','data_contract_revision','time_feature_revision','metrics_revision','prediction_contract_revision','time_feature_fingerprint']
    # Backbone identity names shared contract as shared_contract_version; ExperimentCommon uses contract_version.
    for k in keys:
        a=rid.get(k); b=cache_ident.get('contract_version') if k=='shared_contract_version' else cache_ident.get(k)
        if a is not None and b is not None and str(a)!=str(b): raise RuntimeError(f'Backbone/cache identity mismatch {k}: {a} != {b}')
