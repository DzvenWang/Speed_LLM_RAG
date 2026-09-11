from __future__ import annotations

from pathlib import Path
import shutil
import numpy as np
import pandas as pd

from .sampler import all_unique_group_pick, balanced_unique_group_pick, quotas_for_total
from .utils import canonical_sha256, canonical_view_hash, ensure_dir, read_json, sha256_file, write_json


def challenge_pool_ids(cfg: dict, split: str) -> np.ndarray:
    p = Path(cfg['project']['support_artifact_root']) / cfg['experiment']['dataset'] / f'{split}_challenge_pool.npz'
    if not p.exists():
        raise FileNotFoundError(p)
    return np.load(p, allow_pickle=False)['view_id'].astype(str)


def filtered_views(bench, cfg: dict, split: str, challenge_only: bool) -> pd.DataFrame:
    ec = cfg['experiment']
    fams = list(ec['families'])
    severity = float(ec['severity'])
    if challenge_only:
        ids = set(challenge_pool_ids(cfg, split).tolist())
        base = bench.views[bench.views.view_id.astype(str).isin(ids)].copy()
    else:
        base = bench.views_for(split).copy()
    base = base[base.family.astype(str).isin(fams)]
    base = base[np.isclose(base.severity.astype(float), severity)]
    if 'distribution' in base.columns:
        base = base[base.distribution.astype(str).eq(str(ec['distribution']))]
    return base.reset_index(drop=True)


def _canonicalize(df: pd.DataFrame, stage: str, split: str) -> pd.DataFrame:
    cols = ['view_id', 'view_group_id', 'split', 'window_start', 'family', 'severity', 'distribution']
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f'SupportBench view manifest missing columns: {missing}')
    out = df[cols].copy().reset_index(drop=True)
    out.insert(0, 'protocol_view_id', [f'{stage}_{split}_{i:08d}' for i in range(len(out))])
    return out


def materialize_eval_masks(bench, views: pd.DataFrame, path: Path) -> dict:
    M, P, N = len(views), int(bench.P), int(bench.N)
    direct = np.zeros((M, N), dtype=np.uint8)
    support = np.zeros((M, N), dtype=np.uint8)
    future_valid = np.zeros((M, P, N), dtype=np.uint8)
    view_ids = np.empty(M, dtype=f'<U{max(1, max(len(str(v)) for v in views.view_id.astype(str)))}')
    for i, vid in enumerate(views.view_id.astype(str)):
        s = bench.sample(vid)
        direct[i] = s.artificial_mask.any(axis=0).astype(np.uint8)
        support[i] = np.asarray(bench.predictive_context_metrics(s)['support_dependent_target_mask'], dtype=np.uint8)
        future_valid[i] = s.raw_valid_future.astype(np.uint8)
        view_ids[i] = vid
        if (i + 1) % 500 == 0:
            print(f'evaluation masks: {i+1}/{M}', flush=True)
    np.savez_compressed(path, view_id=view_ids, direct_mask=direct, support_mask=support, future_valid=future_valid)
    return {
        'rows': M,
        'direct_target_fraction': float(direct.mean()),
        'support_target_fraction': float(support.mean()),
        'future_valid_fraction': float(future_valid.mean()),
        'file_sha256': sha256_file(path),
    }


def _split_summary(df: pd.DataFrame, fams: list[str]) -> dict:
    counts = {f: int((df.family.astype(str) == f).sum()) for f in fams}
    return {
        'rows': int(len(df)),
        'unique_groups': int(df.view_group_id.astype(str).nunique()),
        'families': counts,
        'view_hash': canonical_view_hash(df),
    }


def build_stage(bench, cfg: dict, stage: str, force: bool = False) -> Path:
    artifact_root = Path(cfg['project']['artifact_root'])
    dataset = cfg['experiment']['dataset']
    protocol_name = cfg['protocol']['name']
    root = artifact_root / dataset / protocol_name
    sd = root / stage
    if (sd / 'FROZEN.json').exists() and not force:
        raise RuntimeError(f'Refusing to overwrite frozen protocol stage: {sd}')
    if force and sd.exists():
        shutil.rmtree(sd)
    ensure_dir(sd)

    ec = cfg['experiment']
    fams = list(ec['families'])
    seed = int(ec['seed'])

    if stage == 'development':
        train = balanced_unique_group_pick(
            filtered_views(bench, cfg, 'train', False), fams,
            int(cfg['stages']['development']['train_size']), seed,
        )
        val = balanced_unique_group_pick(
            filtered_views(bench, cfg, 'val', True), fams,
            int(cfg['stages']['development']['val_size']), seed + 1,
        )
        split_map = {'train': _canonicalize(train, stage, 'train'), 'val': _canonicalize(val, stage, 'val')}
    elif stage == 'formal':
        dev = root / 'development'
        if not (dev / 'FROZEN.json').exists():
            raise RuntimeError('Development protocol must be frozen before formal protocol.')
        train = all_unique_group_pick(filtered_views(bench, cfg, 'train', False), fams, seed + 10)
        val = pd.read_csv(dev / 'val_views.csv')
        # Formal Val is exactly Development Val; only protocol_view_id namespace changes.
        val2 = val.drop(columns=['protocol_view_id']).copy()
        val2.insert(0, 'protocol_view_id', [f'formal_val_{i:08d}' for i in range(len(val2))])
        split_map = {'train': _canonicalize(train, stage, 'train'), 'val': val2}
    elif stage == 'evaluation':
        auth = root / 'EVALUATION_AUTHORIZED.json'
        if not auth.exists():
            raise RuntimeError(
                'Evaluation protocol is locked. Run scripts/authorize_evaluation.py only after all method configurations are frozen.'
            )
        test = balanced_unique_group_pick(
            filtered_views(bench, cfg, 'test', True), fams,
            int(cfg['stages']['evaluation']['test_size']), seed + 20,
        )
        split_map = {'test': _canonicalize(test, stage, 'test')}
    else:
        raise ValueError(stage)

    for split, df in split_map.items():
        df.to_csv(sd / f'{split}_views.csv', index=False)

    speed_name = bench.manifest['primary_channel']
    st = bench.stats['channels'][speed_name]
    source_info = {
        'supportbench_package_version': cfg['source']['supportbench_version'],
        'supportbench_manifest_sha256': canonical_sha256(bench.manifest),
        'supportbench_view_manifest_sha256': sha256_file(bench.sg_dir / 'views' / 'view_manifest.csv'),
        'challenge_pool_sha256': {},
    }
    for split in ('val', 'test'):
        p = Path(cfg['project']['support_artifact_root']) / dataset / f'{split}_challenge_pool.npz'
        if p.exists():
            source_info['challenge_pool_sha256'][split] = sha256_file(p)

    summary = {
        'protocol_version': cfg['version'],
        'protocol_name': protocol_name,
        'stage': stage,
        'dataset': dataset,
        'history': int(bench.H),
        'horizon': int(bench.P),
        'num_nodes': int(bench.N),
        'severity': float(ec['severity']),
        'distribution': str(ec['distribution']),
        'seed': seed,
        'families': fams,
        'splits': {k: _split_summary(v, fams) for k, v in split_map.items()},
        'normalization': {
            'policy': 'train_only_zscore',
            'primary_channel': speed_name,
            'mean': float(st['mean']),
            'std': float(st['std']),
            'normalized_missing_fill_value': float(cfg['protocol']['normalized_missing_fill_value']),
        },
        'source': source_info,
        'config_sha256': sha256_file(Path(cfg['project']['protocol_root']) / 'configs' / 'protocol.yaml'),
    }

    # Evaluation masks are frozen by Protocol, not recomputed independently by consumers.
    if stage in ('development', 'formal'):
        mask_info = materialize_eval_masks(bench, split_map['val'], sd / 'val_evaluation_masks.npz')
        summary['evaluation_masks'] = {'val': mask_info}
        if stage == 'formal':
            dev_manifest = read_json(root / 'development' / 'stage_manifest.json')
            # Ignore protocol_view_id namespace: source view hashes must be exactly identical.
            dev_val = pd.read_csv(root / 'development' / 'val_views.csv').drop(columns=['protocol_view_id'])
            formal_val = split_map['val'].drop(columns=['protocol_view_id'])
            if canonical_sha256(dev_val.astype(str).to_dict(orient='records')) != canonical_sha256(formal_val.astype(str).to_dict(orient='records')):
                raise AssertionError('Formal Val does not exactly reuse Development Val.')
            summary['reuses_development_val_stage_hash'] = dev_manifest['stage_hash']
    else:
        summary['evaluation_masks'] = {'test': materialize_eval_masks(bench, split_map['test'], sd / 'test_evaluation_masks.npz')}

    summary['stage_hash'] = canonical_sha256({k: v for k, v in summary.items() if k != 'stage_hash'})
    write_json(sd / 'stage_manifest.json', summary)
    write_json(sd / 'FROZEN.json', {
        'protocol_name': protocol_name,
        'protocol_version': cfg['version'],
        'stage': stage,
        'stage_hash': summary['stage_hash'],
        'immutable': True,
    })

    root_manifest = {
        'protocol_name': protocol_name,
        'protocol_version': cfg['version'],
        'dataset': dataset,
        'stages': {},
    }
    rp = root / 'protocol_manifest.json'
    if rp.exists():
        root_manifest = read_json(rp)
    root_manifest.setdefault('stages', {})[stage] = {
        'stage_hash': summary['stage_hash'],
        'splits': summary['splits'],
    }
    root_manifest['protocol_hash'] = canonical_sha256(root_manifest['stages'])
    write_json(rp, root_manifest)
    return sd


def validate_stage(cfg: dict, stage: str) -> dict:
    from .consumer import load_frozen_stage

    dataset = cfg['experiment']['dataset']
    protocol_name = cfg['protocol']['name']
    manifest, views = load_frozen_stage(cfg['project']['artifact_root'], dataset, protocol_name, stage)
    fams = list(cfg['experiment']['families'])
    expected = cfg['stages'][stage]
    for split, df in views.items():
        if df.view_group_id.astype(str).duplicated().any():
            raise AssertionError(f'{stage}/{split}: duplicate view_group_id')
        if set(df.family.astype(str)) - set(fams):
            raise AssertionError(f'{stage}/{split}: unexpected family')
        if split in ('train', 'val', 'test'):
            configured = expected.get(f'{split}_size', None)
            if configured not in (None, 'all', 0) and len(df) != int(configured):
                raise AssertionError(f'{stage}/{split}: expected {configured}, got {len(df)}')
        counts = df.family.astype(str).value_counts().to_dict()
        require_exact_balance = not (stage == 'formal' and split == 'train')
        if require_exact_balance and max(counts.values()) - min(counts.values()) > 1:
            raise AssertionError(f'{stage}/{split}: family counts not balanced: {counts}')
        if stage == 'formal' and split == 'train':
            missing_families = [f for f in fams if int(counts.get(f, 0)) == 0]
            if missing_families:
                raise AssertionError(f'{stage}/{split}: missing sensing families: {missing_families}')
    if 'train' in views and 'val' in views:
        if set(views['train'].view_id.astype(str)) & set(views['val'].view_id.astype(str)):
            raise AssertionError('Train/Val view_id overlap')
        if set(views['train'].view_group_id.astype(str)) & set(views['val'].view_group_id.astype(str)):
            raise AssertionError('Train/Val view_group_id overlap')
    return manifest
