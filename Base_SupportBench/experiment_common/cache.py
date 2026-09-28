from __future__ import annotations

from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd

from .constants import (
    CONTRACT_VERSION, CACHE_SCHEMA_VERSION, DATA_CONTRACT_REVISION,
    TIME_FEATURE_REVISION, METRICS_REVISION, PREDICTION_CONTRACT_REVISION,
)
from .time_features import resolve_time_features, assert_resolved_time_features, audit_cache_time_features
from .utils import canonical_sha256, manifest_identity, write_json

STAGES = ('development', 'formal', 'evaluation')
DEFAULT_FAMILIES = (
    'entry_random',
    'sensor_outage',
    'temporal_outage',
    'regional_outage',
    'directional_outage',
)


def _severity_tag(value: float) -> str:
    return f's{int(round(float(value) * 100)):02d}'


def _allowed_stages(cfg: dict) -> set[str]:
    return {str(x) for x in cfg.get('protocol', {}).get('allowed_stages', STAGES)}


def _assert_stage_allowed(cfg: dict, stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(stage)
    allowed = _allowed_stages(cfg)
    if stage not in allowed:
        raise RuntimeError(
            f"Stage {stage!r} is disabled for {cfg['protocol']['name']}; "
            f'allowed_stages={sorted(allowed)}'
        )


def protocol_stage_dir(cfg: dict, stage: str) -> Path:
    _assert_stage_allowed(cfg, stage)
    return (
        Path(cfg['project']['protocol_artifact_root'])
        / str(cfg['experiment']['subgraph'])
        / str(cfg['protocol']['name'])
        / stage
    )


def load_protocol_stage_metadata(cfg: dict, stage: str) -> dict:
    """Read immutable Protocol metadata without bypassing evaluation data access.

    This function reads only FROZEN.json/stage_manifest.json so cache identity can be
    resolved before materialization. Actual evaluation rows remain protected by the
    Protocol consumer access gate in prepare_cache/validate_cache.
    """
    sd = protocol_stage_dir(cfg, stage)
    frozen_path = sd / 'FROZEN.json'
    manifest_path = sd / 'stage_manifest.json'
    if not frozen_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f'Frozen Protocol stage not found: {sd}')
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if str(frozen.get('stage_hash')) != str(manifest.get('stage_hash')):
        raise RuntimeError(f'Protocol freeze hash mismatch: {sd}')
    if str(manifest.get('protocol_name')) != str(cfg['protocol']['name']):
        raise RuntimeError('Protocol name mismatch between config and frozen manifest')
    if str(manifest.get('protocol_version')) != str(cfg['protocol']['version']):
        raise RuntimeError(
            f"Protocol version mismatch frozen={manifest.get('protocol_version')} "
            f"required={cfg['protocol']['version']}"
        )
    if not np.isclose(float(manifest.get('severity')), float(cfg['experiment']['severity'])):
        raise RuntimeError(
            f"Protocol severity mismatch frozen={manifest.get('severity')} "
            f"required={cfg['experiment']['severity']}"
        )
    expected_sb = str(cfg.get('source', {}).get('supportbench_benchmark_version', '')).strip()
    if expected_sb:
        actual_sb = str(manifest.get('source', {}).get('supportbench_benchmark_version', ''))
        if actual_sb and actual_sb != expected_sb:
            raise RuntimeError(
                f'Protocol is bound to SupportBench benchmark {actual_sb}, expected {expected_sb}'
            )
    return manifest


def evaluation_access_granted(cfg: dict) -> bool:
    """Protocol v2.1 uses direct access to frozen evaluation caches."""
    return True


def require_evaluation_cache_access(cfg: dict, stage: str) -> None:
    """Backward-compatible no-op: there is no Test access grant file in v2.1."""
    return None


def cache_dir(cfg: dict, stage: str, protocol_stage_hash: str | None = None) -> Path:
    """Return a collision-proof cache path bound to one frozen Protocol stage.

    Identity encoded in the path:
      dataset / protocol_name / severity / stage / full_protocol_stage_hash
    """
    _assert_stage_allowed(cfg, stage)
    if protocol_stage_hash is None:
        protocol_stage_hash = str(load_protocol_stage_metadata(cfg, stage)['stage_hash'])
    base = Path(cfg['project']['cache_root'])
    return (
        base
        / str(cfg['experiment']['subgraph'])
        / str(cfg['protocol']['name'])
        / _severity_tag(float(cfg['experiment']['severity']))
        / stage
        / str(protocol_stage_hash)
    )


def _contract_payload(cfg: dict) -> dict:
    pcfg = cfg['protocol']
    return {
        'contract_version': CONTRACT_VERSION,
        'cache_schema_version': CACHE_SCHEMA_VERSION,
        'data_contract_revision': DATA_CONTRACT_REVISION,
        'time_feature_revision': TIME_FEATURE_REVISION,
        'metrics_revision': METRICS_REVISION,
        'prediction_contract_revision': PREDICTION_CONTRACT_REVISION,
        'subgraph': cfg['experiment']['subgraph'],
        'target_name': str(cfg['experiment'].get('target','speed')),
        'target_channel_index': int(cfg['experiment'].get('target_channel_index',-1)),
        'history': int(cfg['experiment']['history']),
        'horizon': int(cfg['experiment']['horizon']),
        'severity': float(cfg['experiment']['severity']),
        'normalized_fill_value': float(cfg['experiment']['normalized_fill_value']),
        'families': list(cfg['experiment'].get('families', DEFAULT_FAMILIES)),
        'protocol_name': pcfg['name'],
        'protocol_version': str(pcfg['version']),
        'paired_group_id': str(pcfg.get('paired_group_id', '')),
        'allowed_stages': sorted(_allowed_stages(cfg)),
        'stage_assertions': pcfg.get('stage_assertions', {}),
        'time_features': dict(cfg['time_features']),
        'materializer_version': 3,
        'target_name': str(cfg['experiment'].get('target', 'speed')),
        'target_channel_index': int(cfg['experiment'].get('target_channel_index', -1)),
    }


def contract_fingerprint(cfg: dict) -> str:
    return canonical_sha256(_contract_payload(cfg))


def load_cache_identity(path: str | Path) -> dict:
    p = Path(path) / 'cache_manifest.json'
    if not p.exists():
        raise FileNotFoundError(p)
    m = json.loads(p.read_text(encoding='utf-8'))
    required = [
        'contract_version','cache_schema_version','data_contract_revision','time_feature_revision',
        'metrics_revision','prediction_contract_revision','stage','subgraph','protocol_name',
        'protocol_version','severity','protocol_stage_hash','split_protocol_view_hashes',
        'cache_fingerprint','contract_fingerprint','time_feature_fingerprint',
        'time_feature_source','time_feature_parser',
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
        'target_name': str(m.get('target_name','speed')),
        'target_channel_index': int(m.get('target_channel_index',2)),
        'protocol_name': str(m['protocol_name']),
        'protocol_version': str(m['protocol_version']),
        'severity': float(m['severity']),
        'paired_group_id': str(m.get('paired_group_id', '')),
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
        x.get('stage'), x.get('dataset'), x.get('target_name'), x.get('target_channel_index'), x.get('protocol_name'), x.get('protocol_version'),
        x.get('severity'), x.get('paired_group_id'), x.get('protocol_stage_hash'),
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
        'dataset': str(cfg['experiment']['subgraph']),
        'target_name': str(cfg['experiment'].get('target','speed')),
        'target_channel_index': int(cfg['experiment'].get('target_channel_index',-1)),
        'protocol_name': str(cfg['protocol']['name']),
        'protocol_version': str(cfg['protocol']['version']),
        'paired_group_id': str(cfg['protocol'].get('paired_group_id', '')),
    }
    for k,v in expected.items():
        if str(identity.get(k)) != str(v):
            raise RuntimeError(f'Shared cache {k}={identity.get(k)} != required {v}')
    if not np.isclose(float(identity.get('severity')), float(cfg['experiment']['severity'])):
        raise RuntimeError(
            f"Shared cache severity={identity.get('severity')} != required {cfg['experiment']['severity']}"
        )
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


def _assert_split_contract(cfg: dict, stage: str, sets: dict[str, pd.DataFrame]) -> None:
    asserts = cfg.get('protocol', {}).get('stage_assertions', {}).get(stage, {})
    families = list(cfg.get('experiment', {}).get('families', DEFAULT_FAMILIES))
    for split in ('train','val','test'):
        if split not in sets:
            continue
        df = sets[split]
        rows_key = f'{split}_rows'
        if rows_key in asserts and len(df) != int(asserts[rows_key]):
            raise RuntimeError(
                f'Protocol assertion failed {stage}/{split}: expected rows={asserts[rows_key]}, got={len(df)}'
            )
        ug_key = f'{split}_unique_groups'
        if ug_key in asserts:
            got = int(df.view_group_id.astype(str).nunique())
            if got != int(asserts[ug_key]):
                raise RuntimeError(
                    f'Protocol assertion failed {stage}/{split}: expected unique_groups={asserts[ug_key]}, got={got}'
                )
        target_per_family = asserts.get('family_rows', {}).get(split)
        if target_per_family is not None:
            counts = {str(k): int(v) for k,v in df.family.astype(str).value_counts().to_dict().items()}
            bad = {f: counts.get(f, 0) for f in families if counts.get(f, 0) != int(target_per_family)}
            if bad:
                raise RuntimeError(
                    f'Protocol assertion failed {stage}/{split}: expected {target_per_family} per family; got={counts}'
                )
        sev = np.asarray(df.severity, dtype=float)
        if not np.allclose(sev, float(cfg['experiment']['severity'])):
            raise RuntimeError(
                f'Protocol assertion failed {stage}/{split}: mixed/wrong severity for '
                f"{cfg['protocol']['name']}"
            )
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
    target_name = str(bench.manifest['primary_channel'])
    target_idx = int(bench.target_idx)
    mean = float(bench.stats['channels'][target_name]['mean'])
    std = max(float(bench.stats['channels'][target_name]['std']), 1e-6)
    tod = np.asarray(tf.tod, np.int16); dow = np.asarray(tf.dow, np.int8)
    rows = []
    for i, row in enumerate(df.itertuples(index=False)):
        s = bench.sample(str(row.view_id))
        full = s.history_full[..., target_idx].astype(np.float32)
        fulln = (full - mean) / std
        obs = s.observation_mask[..., target_idx].astype(bool)
        art = s.artificial_mask.astype(bool)
        x = np.where(obs, fulln, 0.0).astype(np.float32)
        arr['x_obs_norm'][i] = x
        arr['x_full_norm'][i] = fulln
        arr['history_observed'][i] = obs.astype(np.uint8)
        arr['artificial_missing'][i] = art.astype(np.uint8)
        arr['y_raw'][i] = s.future_target
        arr['y_norm'][i] = (s.future_target - mean) / std
        arr['future_valid'][i] = s.raw_valid_future.astype(np.uint8)
        arr['direct_mask'][i] = art.any(axis=0).astype(np.uint8)
        arr['support_mask'][i] = np.zeros((N,), dtype=np.uint8)  # compatibility only; Base_SupportBench has no RAG support scope
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
        'mean': mean, 'std': std, 'target_name': target_name, 'target_channel_index': target_idx, 'target_mean': mean, 'target_std': std,
        **({'speed_mean': mean, 'speed_std': std} if target_name == 'speed' else {}),
    }


def prepare_cache(cfg: dict, stage: str, *, force: bool = False, allow_locked_evaluation_materialization: bool = False) -> Path:
    _assert_stage_allowed(cfg, stage)
    # Delayed imports keep Base_SupportBench cache layer unit-testable outside the full repo.
    from shared_protocol.consumer import load_frozen_stage
    from supportbench.dataset import SupportBench

    bypass_eval_gate = bool(stage == 'evaluation' and allow_locked_evaluation_materialization)
    pm, sets = load_frozen_stage(
        cfg['project']['protocol_artifact_root'],
        cfg['experiment']['subgraph'],
        cfg['protocol']['name'],
        stage,
        require_evaluation_access=not bypass_eval_gate,
    )
    if str(pm.get('protocol_version')) != str(cfg['protocol']['version']):
        raise RuntimeError(
            f"Frozen Protocol version={pm.get('protocol_version')} != required {cfg['protocol']['version']}"
        )
    if not np.isclose(float(pm.get('severity')), float(cfg['experiment']['severity'])):
        raise RuntimeError(
            f"Frozen Protocol severity={pm.get('severity')} != required {cfg['experiment']['severity']}"
        )
    if stage == 'evaluation':
        pair = pm.get('paired_evaluation', {})
        if str(pair.get('paired_group_id', '')) != str(cfg['protocol'].get('paired_group_id', '')):
            raise RuntimeError('Frozen paired_group_id does not match Base_SupportBench cache layer contract')

    _assert_split_contract(cfg, stage, sets)
    out = cache_dir(cfg, stage, protocol_stage_hash=str(pm['stage_hash']))
    if (out/'READY.json').exists() and not force:
        ident = load_cache_identity(out)
        require_contract(cfg, ident, stage)
        if ident['protocol_stage_hash'] != str(pm['stage_hash']):
            raise RuntimeError('Existing cache points to a different frozen Protocol stage')
        print('shared cache already ready:', out, flush=True)
        return out
    out.mkdir(parents=True, exist_ok=True)
    if force:
        _remove_outputs(out)
        out.mkdir(parents=True, exist_ok=True)

    bench = SupportBench(cfg['experiment']['subgraph'], root=cfg['project']['benchmark_root'])
    if int(bench.H) != int(cfg['experiment']['history']) or int(bench.P) != int(cfg['experiment']['horizon']):
        raise RuntimeError(
            f'SupportBench H/P={bench.H}/{bench.P} != contract '
            f'{cfg["experiment"]["history"]}/{cfg["experiment"]["horizon"]}'
        )
    tc = cfg['time_features']
    tf = resolve_time_features(
        bench,
        allow_index_fallback=bool(tc.get('allow_index_modulo_fallback', False)),
        index_origin_dow=int(tc.get('index_origin_dow',0)),
    )
    tf_report = assert_resolved_time_features(tf)
    if tf.freq_minutes != int(tc['expected_freq_minutes']) or tf.steps_per_day != int(tc['expected_steps_per_day']):
        raise RuntimeError('Resolved time grid does not match shared contract')
    print(
        f'time features PASS source={tf.source} parser={tf.parser} '
        f'TOD={tf_report["tod_unique"]} DOW={tf_report["dow_unique"]} '
        f'fingerprint={tf_report["fingerprint"]}',
        flush=True,
    )

    split_info = {}
    for split, df in sets.items():
        print(
            f'{stage}/{split} <= {cfg["protocol"]["name"]}: {len(df)} views '
            f'unique_groups={df.view_group_id.astype(str).nunique()} '
            f'families={df.family.value_counts().to_dict()}',
            flush=True,
        )
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
        'stage': stage,
        'subgraph': cfg['experiment']['subgraph'],
        'target_name': str(cfg['experiment'].get('target','speed')),
        'target_channel_index': int(cfg['experiment'].get('target_channel_index',-1)),
        'history': int(cfg['experiment']['history']),
        'horizon': int(cfg['experiment']['horizon']),
        'severity': float(cfg['experiment']['severity']),
        'protocol_name': cfg['protocol']['name'],
        'protocol_version': str(cfg['protocol']['version']),
        'paired_group_id': str(cfg['protocol'].get('paired_group_id', '')),
        'protocol_stage_hash': str(pm['stage_hash']),
        'split_protocol_view_hashes': {k: str(pm['splits'][k]['view_hash']) for k in sets},
        'splits': split_info,
        'time_feature_source': tf.source,
        'time_feature_column': tf.column,
        'time_feature_parser': tf.parser,
        'time_feature_freq_minutes': int(tf.freq_minutes),
        'time_feature_steps_per_day': int(tf.steps_per_day),
        'time_feature_fingerprint': tf_report['fingerprint'],
        'time_feature_cache_audit_status': tf_cache['status'],
    }
    if stage == 'evaluation':
        summary['paired_evaluation'] = dict(pm.get('paired_evaluation', {}))
        granted = evaluation_access_granted(cfg)
        summary['evaluation_access_at_materialization'] = 'GRANTED' if granted else 'LOCKED'
        summary['evaluation_cache_materialized_while_locked'] = bool(not granted)
    summary['cache_fingerprint'] = canonical_sha256(summary)
    write_json(out/'cache_manifest.json', summary)
    write_json(out/'READY.json', {
        'contract_version': CONTRACT_VERSION,
        'stage': stage,
        'status': 'READY',
        'dataset': cfg['experiment']['subgraph'],
        'protocol_name': cfg['protocol']['name'],
        'severity': float(cfg['experiment']['severity']),
        'protocol_stage_hash': str(pm['stage_hash']),
        'cache_fingerprint': summary['cache_fingerprint'],
        'time_feature_fingerprint': summary['time_feature_fingerprint'],
        'evaluation_access_state': ('GRANTED' if evaluation_access_granted(cfg) else 'LOCKED') if stage == 'evaluation' else 'N/A',
        'materialized_while_locked': bool(stage == 'evaluation' and not evaluation_access_granted(cfg)),
    })
    print('SHARED_CACHE=' + str(out), flush=True)
    print('CACHE_FINGERPRINT=' + summary['cache_fingerprint'], flush=True)
    return out


def validate_cache(cfg: dict, stage: str, *, allow_locked_evaluation_validation: bool = False) -> dict:
    _assert_stage_allowed(cfg, stage)
    from shared_protocol.consumer import load_frozen_stage
    from supportbench.dataset import SupportBench

    bypass_eval_gate = bool(stage == 'evaluation' and allow_locked_evaluation_validation)
    pm, pviews = load_frozen_stage(
        cfg['project']['protocol_artifact_root'],
        cfg['experiment']['subgraph'],
        cfg['protocol']['name'],
        stage,
        require_evaluation_access=not bypass_eval_gate,
    )
    _assert_split_contract(cfg, stage, pviews)
    out = cache_dir(cfg, stage, protocol_stage_hash=str(pm['stage_hash']))
    mf = out/'cache_manifest.json'
    ready = out/'READY.json'
    if not mf.exists() or not ready.exists():
        raise RuntimeError(f'Missing cache manifest/READY in {out}')
    j = json.loads(mf.read_text(encoding='utf-8'))
    r = json.loads(ready.read_text(encoding='utf-8'))
    fp = j.get('cache_fingerprint')
    tmp = dict(j)
    tmp.pop('cache_fingerprint', None)
    if canonical_sha256(tmp) != fp or r.get('cache_fingerprint') != fp:
        raise RuntimeError('Shared cache fingerprint/READY mismatch')
    ident = load_cache_identity(out)
    require_contract(cfg, ident, stage)
    if str(j['protocol_stage_hash']) != str(pm['stage_hash']):
        raise RuntimeError('Shared cache Protocol stage hash mismatch')

    expected = {'development': {'train','val'}, 'formal': {'train','val'}, 'evaluation': {'test'}}[stage]
    if set(j['splits']) != expected:
        raise RuntimeError(f'Split isolation mismatch expected={expected} actual={set(j["splits"])}')

    bench = SupportBench(cfg['experiment']['subgraph'], root=cfg['project']['benchmark_root'])
    tc = cfg['time_features']
    tf = resolve_time_features(
        bench,
        allow_index_fallback=bool(tc.get('allow_index_modulo_fallback',False)),
        index_origin_dow=int(tc.get('index_origin_dow',0)),
    )
    tfr = assert_resolved_time_features(tf)
    for key,val in [
        ('time_feature_fingerprint',tfr['fingerprint']),
        ('time_feature_source',tf.source),
        ('time_feature_parser',tf.parser),
    ]:
        if str(j.get(key)) != str(val):
            raise RuntimeError(f'{key} mismatch cache={j.get(key)} current={val}')
    tfa = audit_cache_time_features(out, steps_per_day=tf.steps_per_day, expected_splits=expected)
    if tfa['status'] != 'PASS':
        raise RuntimeError('Shared cache time-feature audit failed: ' + json.dumps(tfa['errors']))

    H = int(cfg['experiment']['history'])
    P = int(cfg['experiment']['horizon'])
    required_arrays = [
        'x_obs_norm','x_full_norm','history_observed','artificial_missing','y_raw','y_norm',
        'future_valid','direct_mask','support_mask','tod_hist','dow_hist','tod_future','dow_future',
    ]
    for split in expected:
        meta = pd.read_csv(out/f'{split}_manifest.csv')
        proto = pviews[split]
        cols = ['protocol_view_id','view_id','view_group_id','window_start','family','severity','distribution']
        if not meta[cols].astype(str).reset_index(drop=True).equals(proto[cols].astype(str).reset_index(drop=True)):
            raise RuntimeError(f'{stage}/{split} differs from frozen Protocol')
        if manifest_identity(meta) != j['splits'][split]['view_manifest_sha256']:
            raise RuntimeError(f'{split} view-manifest fingerprint mismatch')
        for name in required_arrays:
            p = out/f'{split}_{name}.npy'
            if not p.exists():
                raise RuntimeError(f'Missing {p}')
        x = np.load(out/f'{split}_x_obs_norm.npy', mmap_mode='r')
        y = np.load(out/f'{split}_y_raw.npy', mmap_mode='r')
        if x.shape != (len(meta),H,int(bench.N)) or y.shape != (len(meta),P,int(bench.N)):
            raise RuntimeError(f'{split} tensor shape mismatch x={x.shape} y={y.shape}')
    if not (out/'adjacency.npy').exists():
        raise RuntimeError('Missing adjacency.npy')
    return ident
