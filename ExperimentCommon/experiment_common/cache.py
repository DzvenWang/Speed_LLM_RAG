from __future__ import annotations
from pathlib import Path
import json, shutil
import numpy as np
import pandas as pd

from .constants import (
    CONTRACT_VERSION, CACHE_SCHEMA_VERSION, DATA_CONTRACT_REVISION,
    TIME_FEATURE_REVISION, METRICS_REVISION, PREDICTION_CONTRACT_REVISION,
)
from .time_features import resolve_time_features, assert_resolved_time_features, audit_cache_time_features
from .utils import canonical_sha256, manifest_identity, write_json

STAGES = ('development', 'formal', 'evaluation')


def cache_dir(cfg: dict, stage: str) -> Path:
    if stage not in STAGES:
        raise ValueError(stage)
    return Path(cfg['project']['cache_root']) / cfg['experiment']['subgraph'] / stage


def _contract_payload(cfg: dict) -> dict:
    return {
        'contract_version': CONTRACT_VERSION,
        'cache_schema_version': CACHE_SCHEMA_VERSION,
        'data_contract_revision': DATA_CONTRACT_REVISION,
        'time_feature_revision': TIME_FEATURE_REVISION,
        'metrics_revision': METRICS_REVISION,
        'prediction_contract_revision': PREDICTION_CONTRACT_REVISION,
        'subgraph': cfg['experiment']['subgraph'],
        'history': int(cfg['experiment']['history']),
        'horizon': int(cfg['experiment']['horizon']),
        'normalized_fill_value': float(cfg['experiment']['normalized_fill_value']),
        'protocol_name': cfg['protocol']['name'],
        'protocol_version': str(cfg['protocol']['version']),
        'time_features': dict(cfg['time_features']),
        'materializer_version': 1,
    }


def contract_fingerprint(cfg: dict) -> str:
    return canonical_sha256(_contract_payload(cfg))


def load_cache_identity(path: str | Path) -> dict:
    p = Path(path) / 'cache_manifest.json'
    if not p.exists():
        raise FileNotFoundError(p)
    m = json.loads(p.read_text())
    required = [
        'contract_version','cache_schema_version','data_contract_revision','time_feature_revision',
        'metrics_revision','prediction_contract_revision','stage','subgraph','protocol_stage_hash',
        'split_protocol_view_hashes','cache_fingerprint','contract_fingerprint',
        'time_feature_fingerprint','time_feature_source','time_feature_parser',
    ]
    missing = [k for k in required if k not in m]
    if missing:
        raise RuntimeError(f'Shared cache missing contract fields: {missing}')
    return {
        'contract_version': str(m['contract_version']),
        'cache_schema_version': int(m['cache_schema_version']),
        'data_contract_revision': str(m['data_contract_revision']),
        'time_feature_revision': str(m['time_feature_revision']),
        'metrics_revision': str(m['metrics_revision']),
        'prediction_contract_revision': str(m['prediction_contract_revision']),
        'stage': str(m['stage']),
        'dataset': str(m['subgraph']),
        'protocol_stage_hash': str(m['protocol_stage_hash']),
        'split_view_hashes': {str(k): str(v) for k,v in m['split_protocol_view_hashes'].items()},
        'cache_fingerprint': str(m['cache_fingerprint']),
        'contract_fingerprint': str(m['contract_fingerprint']),
        'time_feature_fingerprint': str(m['time_feature_fingerprint']),
        'time_feature_source': str(m['time_feature_source']),
        'time_feature_parser': str(m['time_feature_parser']),
    }


def identity_key(x: dict) -> tuple:
    return (
        x.get('contract_version'), x.get('cache_schema_version'), x.get('data_contract_revision'),
        x.get('time_feature_revision'), x.get('metrics_revision'), x.get('prediction_contract_revision'),
        x.get('stage'), x.get('dataset'), x.get('protocol_stage_hash'),
        json.dumps(x.get('split_view_hashes', {}), sort_keys=True), x.get('cache_fingerprint'),
        x.get('contract_fingerprint'), x.get('time_feature_fingerprint'), x.get('time_feature_source'),
        x.get('time_feature_parser'),
    )


def require_contract(cfg: dict, identity: dict, stage: str | None = None) -> None:
    expected = {
        'contract_version': CONTRACT_VERSION,
        'cache_schema_version': CACHE_SCHEMA_VERSION,
        'data_contract_revision': DATA_CONTRACT_REVISION,
        'time_feature_revision': TIME_FEATURE_REVISION,
        'metrics_revision': METRICS_REVISION,
        'prediction_contract_revision': PREDICTION_CONTRACT_REVISION,
        'contract_fingerprint': contract_fingerprint(cfg),
    }
    for k,v in expected.items():
        if str(identity.get(k)) != str(v):
            raise RuntimeError(f'Shared cache {k}={identity.get(k)} != required {v}')
    if stage is not None and str(identity.get('stage')) != str(stage):
        raise RuntimeError(f'Shared cache stage={identity.get("stage")} != required {stage}')


def _remove_outputs(out: Path) -> None:
    if not out.exists():
        return
    for p in out.iterdir():
        if p.is_file() or p.is_symlink():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)


def _materialize_split(bench, df: pd.DataFrame, out: Path, split: str, tf) -> dict:
    H, P, N = int(bench.H), int(bench.P), int(bench.N)
    M = int(len(df))
    shapes = {
        'x_obs_norm': ((M,H,N), np.float32),
        'x_full_norm': ((M,H,N), np.float32),
        'history_observed': ((M,H,N), np.uint8),
        'artificial_missing': ((M,H,N), np.uint8),
        'y_raw': ((M,P,N), np.float32),
        'y_norm': ((M,P,N), np.float32),
        'future_valid': ((M,P,N), np.uint8),
        'direct_mask': ((M,N), np.uint8),
        'support_mask': ((M,N), np.uint8),
        'tod_hist': ((M,H), np.int16),
        'dow_hist': ((M,H), np.int8),
        'tod_future': ((M,P), np.int16),
        'dow_future': ((M,P), np.int8),
    }
    arr = {k: np.lib.format.open_memmap(out/f'{split}_{k}.npy', mode='w+', dtype=dt, shape=sh)
           for k,(sh,dt) in shapes.items()}
    speed_name = bench.manifest['primary_channel']
    mean = float(bench.stats['channels'][speed_name]['mean'])
    std = max(float(bench.stats['channels'][speed_name]['std']), 1e-6)
    tod = np.asarray(tf.tod, np.int16); dow = np.asarray(tf.dow, np.int8)
    rows = []
    for i, row in enumerate(df.itertuples(index=False)):
        s = bench.sample(str(row.view_id))
        full = s.history_full[..., bench.speed_idx].astype(np.float32)
        fulln = (full - mean) / std
        obs = s.observation_mask[..., bench.speed_idx].astype(bool)
        art = s.artificial_mask.astype(bool)
        x = np.where(obs, fulln, 0.0).astype(np.float32)
        arr['x_obs_norm'][i] = x
        arr['x_full_norm'][i] = fulln
        arr['history_observed'][i] = obs.astype(np.uint8)
        arr['artificial_missing'][i] = art.astype(np.uint8)
        arr['y_raw'][i] = s.future_speed
        arr['y_norm'][i] = (s.future_speed - mean) / std
        arr['future_valid'][i] = s.raw_valid_future.astype(np.uint8)
        arr['direct_mask'][i] = art.any(axis=0).astype(np.uint8)
        arr['support_mask'][i] = np.asarray(bench.predictive_context_metrics(s)['support_dependent_target_mask'], np.uint8)
        st = int(s.window_start)
        arr['tod_hist'][i] = tod[st:st+H]; arr['dow_hist'][i] = dow[st:st+H]
        arr['tod_future'][i] = tod[st+H:st+H+P]; arr['dow_future'][i] = dow[st+H:st+H+P]
        rows.append({
            'protocol_view_id': getattr(row, 'protocol_view_id', ''),
            'view_id': s.view_id, 'view_group_id': s.view_group_id, 'window_start': st,
            'family': str(s.view_spec['family']), 'severity': float(s.view_spec['severity']),
            'distribution': str(s.view_spec.get('distribution', 'id')),
        })
        if (i+1) % 500 == 0:
            print(f'materialize {split}: {i+1}/{M}', flush=True)
    for a in arr.values():
        a.flush()
    meta = pd.DataFrame(rows)
    meta.to_csv(out/f'{split}_manifest.csv', index=False)
    return {
        'rows': M, 'unique_groups': int(meta.view_group_id.astype(str).nunique()),
        'families': {str(k): int(v) for k,v in meta.family.value_counts().to_dict().items()},
        'view_manifest_sha256': manifest_identity(meta),
        'mean': mean, 'std': std, 'speed_mean': mean, 'speed_std': std,
    }


def prepare_cache(cfg: dict, stage: str, *, force: bool = False) -> Path:
    if stage not in STAGES:
        raise ValueError(stage)
    # Imports are intentionally delayed so ExperimentCommon remains unit-testable off Gadi.
    from shared_protocol.consumer import load_frozen_stage
    from supportbench.dataset import SupportBench

    out = cache_dir(cfg, stage)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'READY.json').exists() and not force:
        ident = load_cache_identity(out); require_contract(cfg, ident, stage)
        print('shared cache already ready:', out, flush=True)
        return out
    if force:
        _remove_outputs(out); out.mkdir(parents=True, exist_ok=True)

    pm, sets = load_frozen_stage(
        cfg['project']['protocol_artifact_root'], cfg['experiment']['subgraph'], cfg['protocol']['name'], stage
    )
    asserts = cfg.get('protocol', {}).get('stage_assertions', {}).get(stage, {})
    for split in ('train','val','test'):
        key=f'{split}_rows'
        if key in asserts and split in sets and len(sets[split]) != int(asserts[key]):
            raise RuntimeError(f'Protocol assertion failed {stage}/{split}: expected {asserts[key]}, got {len(sets[split])}')

    bench = SupportBench(cfg['experiment']['subgraph'], root=cfg['project']['benchmark_root'])
    if int(bench.H) != int(cfg['experiment']['history']) or int(bench.P) != int(cfg['experiment']['horizon']):
        raise RuntimeError(f'SupportBench H/P={bench.H}/{bench.P} != contract {cfg["experiment"]["history"]}/{cfg["experiment"]["horizon"]}')
    tc = cfg['time_features']
    tf = resolve_time_features(
        bench, allow_index_fallback=bool(tc.get('allow_index_modulo_fallback', False)),
        index_origin_dow=int(tc.get('index_origin_dow',0)),
    )
    tf_report = assert_resolved_time_features(tf)
    if tf.freq_minutes != int(tc['expected_freq_minutes']) or tf.steps_per_day != int(tc['expected_steps_per_day']):
        raise RuntimeError('Resolved time grid does not match shared contract')
    print(f'time features PASS source={tf.source} parser={tf.parser} TOD={tf_report["tod_unique"]} DOW={tf_report["dow_unique"]} fingerprint={tf_report["fingerprint"]}', flush=True)

    split_info = {}
    for split, df in sets.items():
        print(f'{stage}/{split} <= SharedProtocol: {len(df)} views unique_groups={df.view_group_id.astype(str).nunique()} families={df.family.value_counts().to_dict()}', flush=True)
        split_info[split] = _materialize_split(bench, df, out, split, tf)
    np.save(out/'adjacency.npy', np.asarray(bench.adj, np.float32))

    expected_splits = {'development': {'train','val'}, 'formal': {'train','val'}, 'evaluation': {'test'}}[stage]
    tf_cache = audit_cache_time_features(out, steps_per_day=tf.steps_per_day, expected_splits=expected_splits)
    if tf_cache['status'] != 'PASS':
        raise RuntimeError('Materialized shared cache failed time-feature audit: ' + json.dumps(tf_cache['errors']))
    write_json(out/'time_feature_audit.json', tf_cache)
    write_json(out/'resolved_time_feature_audit.json', tf_report)

    summary = {
        'contract_version': CONTRACT_VERSION,
        'cache_schema_version': CACHE_SCHEMA_VERSION,
        'data_contract_revision': DATA_CONTRACT_REVISION,
        'time_feature_revision': TIME_FEATURE_REVISION,
        'metrics_revision': METRICS_REVISION,
        'prediction_contract_revision': PREDICTION_CONTRACT_REVISION,
        'contract_fingerprint': contract_fingerprint(cfg),
        'stage': stage, 'subgraph': cfg['experiment']['subgraph'],
        'history': int(cfg['experiment']['history']), 'horizon': int(cfg['experiment']['horizon']),
        'protocol_name': cfg['protocol']['name'], 'protocol_version': str(cfg['protocol']['version']),
        'protocol_stage_hash': pm['stage_hash'],
        'split_protocol_view_hashes': {k: str(pm['splits'][k]['view_hash']) for k in sets},
        'splits': split_info,
        'time_feature_source': tf.source, 'time_feature_column': tf.column, 'time_feature_parser': tf.parser,
        'time_feature_freq_minutes': int(tf.freq_minutes), 'time_feature_steps_per_day': int(tf.steps_per_day),
        'time_feature_fingerprint': tf_report['fingerprint'],
        'time_feature_cache_audit_status': tf_cache['status'],
    }
    summary['cache_fingerprint'] = canonical_sha256(summary)
    write_json(out/'cache_manifest.json', summary)
    write_json(out/'READY.json', {
        'contract_version': CONTRACT_VERSION, 'stage': stage, 'status': 'READY',
        'protocol_stage_hash': pm['stage_hash'], 'cache_fingerprint': summary['cache_fingerprint'],
        'time_feature_fingerprint': summary['time_feature_fingerprint'],
    })
    print('SHARED_CACHE=' + str(out), flush=True)
    print('CACHE_FINGERPRINT=' + summary['cache_fingerprint'], flush=True)
    return out


def validate_cache(cfg: dict, stage: str) -> dict:
    from shared_protocol.consumer import load_frozen_stage
    from supportbench.dataset import SupportBench

    out = cache_dir(cfg, stage)
    mf=out/'cache_manifest.json'; ready=out/'READY.json'
    if not mf.exists() or not ready.exists():
        raise RuntimeError(f'Missing cache manifest/READY in {out}')
    j=json.loads(mf.read_text()); r=json.loads(ready.read_text())
    fp=j.get('cache_fingerprint'); tmp=dict(j); tmp.pop('cache_fingerprint',None)
    if canonical_sha256(tmp) != fp or r.get('cache_fingerprint') != fp:
        raise RuntimeError('Shared cache fingerprint/READY mismatch')
    ident=load_cache_identity(out); require_contract(cfg, ident, stage)
    pm,pviews=load_frozen_stage(cfg['project']['protocol_artifact_root'], cfg['experiment']['subgraph'], cfg['protocol']['name'], stage)
    if j['protocol_stage_hash'] != pm['stage_hash']:
        raise RuntimeError('Shared cache Protocol stage hash mismatch')
    expected={'development': {'train','val'}, 'formal': {'train','val'}, 'evaluation': {'test'}}[stage]
    if set(j['splits']) != expected:
        raise RuntimeError(f'Split isolation mismatch expected={expected} actual={set(j["splits"])}')

    bench=SupportBench(cfg['experiment']['subgraph'], root=cfg['project']['benchmark_root'])
    tc=cfg['time_features']; tf=resolve_time_features(bench, allow_index_fallback=bool(tc.get('allow_index_modulo_fallback',False)), index_origin_dow=int(tc.get('index_origin_dow',0)))
    tfr=assert_resolved_time_features(tf)
    for key,val in [('time_feature_fingerprint',tfr['fingerprint']),('time_feature_source',tf.source),('time_feature_parser',tf.parser)]:
        if str(j.get(key)) != str(val):
            raise RuntimeError(f'{key} mismatch cache={j.get(key)} current={val}')
    tfa=audit_cache_time_features(out, steps_per_day=tf.steps_per_day, expected_splits=expected)
    if tfa['status'] != 'PASS':
        raise RuntimeError('Shared cache time-feature audit failed: '+json.dumps(tfa['errors']))

    H=int(cfg['experiment']['history']); P=int(cfg['experiment']['horizon'])
    required_arrays=['x_obs_norm','x_full_norm','history_observed','artificial_missing','y_raw','y_norm','future_valid','direct_mask','support_mask','tod_hist','dow_hist','tod_future','dow_future']
    for split in expected:
        meta=pd.read_csv(out/f'{split}_manifest.csv'); proto=pviews[split]
        cols=['protocol_view_id','view_id','view_group_id','window_start','family','severity','distribution']
        if not meta[cols].astype(str).reset_index(drop=True).equals(proto[cols].astype(str).reset_index(drop=True)):
            raise RuntimeError(f'{stage}/{split} differs from frozen Shared Protocol')
        if manifest_identity(meta) != j['splits'][split]['view_manifest_sha256']:
            raise RuntimeError(f'{split} view-manifest fingerprint mismatch')
        for name in required_arrays:
            p=out/f'{split}_{name}.npy'
            if not p.exists(): raise RuntimeError(f'Missing {p}')
        x=np.load(out/f'{split}_x_obs_norm.npy',mmap_mode='r'); y=np.load(out/f'{split}_y_raw.npy',mmap_mode='r')
        if x.shape != (len(meta),H,int(bench.N)) or y.shape != (len(meta),P,int(bench.N)):
            raise RuntimeError(f'{split} tensor shape mismatch x={x.shape} y={y.shape}')
    if not (out/'adjacency.npy').exists():
        raise RuntimeError('Missing adjacency.npy')
    return ident
