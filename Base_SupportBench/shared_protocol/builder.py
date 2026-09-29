from __future__ import annotations

from pathlib import Path
import shutil
import numpy as np
import pandas as pd

from .sampler import all_unique_group_pick, balanced_unique_group_pick
from .utils import canonical_sha256, canonical_view_hash, ensure_dir, read_json, sha256_file, write_json


def filtered_views(
    bench,
    cfg: dict,
    split: str,
    challenge_only: bool,
    severity_override: float | None = None,
) -> pd.DataFrame:
    ec = cfg['experiment']
    fams = list(ec['families'])
    severity = float(ec['severity'] if severity_override is None else severity_override)
    if challenge_only:
        raise RuntimeError('Challenge pools are not part of Base_SupportBench.')
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


def _pairing_root(cfg: dict) -> Path:
    group_id = str(cfg['protocol']['paired_group_id'])
    return (
        Path(cfg['project']['artifact_root'])
        / cfg['experiment']['dataset']
        / '_paired'
        / group_id
    )


def _pairing_paths(cfg: dict) -> tuple[Path, Path]:
    root = _pairing_root(cfg)
    return root / 'evaluation_pairs.csv', root / 'PAIRING_FROZEN.json'


def ensure_evaluation_pairing(bench, cfg: dict, force: bool = False) -> tuple[pd.DataFrame, dict]:
    """Create one frozen base-query/family assignment shared by every severity.

    Pairing is sampled from the configured reference severity (normally 40%) using
    the all-eligible ID Test pool. Each member protocol later resolves its own
    severity-specific concrete SupportBench view for the same (view_group_id,
    family), guaranteeing strict paired comparison across severities.
    """
    pair_csv, pair_frozen = _pairing_paths(cfg)
    if pair_csv.exists() and pair_frozen.exists() and not force:
        pair = pd.read_csv(pair_csv)
        frozen = read_json(pair_frozen)
        got = canonical_sha256(pair.astype(str).to_dict(orient='records'))
        if got != frozen.get('pairing_hash'):
            raise RuntimeError(f'Paired evaluation manifest hash mismatch: {pair_csv}')
        return pair, frozen

    if force and pair_csv.parent.exists():
        shutil.rmtree(pair_csv.parent)
    ensure_dir(pair_csv.parent)

    pcfg = cfg['protocol']
    ec = cfg['experiment']
    ref_sev = float(pcfg.get('paired_reference_severity', 0.40))
    fams = list(ec['families'])
    seed = int(ec['seed']) + int(pcfg.get('paired_evaluation_seed_offset', 20))
    total = int(cfg['stages']['evaluation']['test_size'])

    eligible = filtered_views(bench, cfg, 'test', False, severity_override=ref_sev)
    selected = balanced_unique_group_pick(eligible, fams, total, seed)
    keep = ['view_group_id', 'window_start', 'family', 'distribution']
    pair = selected[keep].copy().reset_index(drop=True)
    pair.insert(0, 'pair_id', [f'pair_{i:08d}' for i in range(len(pair))])
    pair['reference_severity'] = ref_sev
    pair.to_csv(pair_csv, index=False)

    counts = pair.family.astype(str).value_counts().to_dict()
    frozen = {
        'paired_group_id': str(pcfg['paired_group_id']),
        'dataset': str(ec['dataset']),
        'reference_severity': ref_sev,
        'distribution': str(ec['distribution']),
        'families': fams,
        'rows': int(len(pair)),
        'family_counts': {f: int(counts.get(f, 0)) for f in fams},
        'pairing_hash': canonical_sha256(pair.astype(str).to_dict(orient='records')),
        'supportbench_view_manifest_sha256': sha256_file(bench.sg_dir / 'views' / 'view_manifest.csv'),
        'immutable': True,
    }
    write_json(pair_frozen, frozen)
    return pair, frozen


def resolve_paired_evaluation_views(bench, cfg: dict, pair: pd.DataFrame) -> pd.DataFrame:
    """Resolve exactly one severity-specific concrete view for every frozen pair."""
    sev = float(cfg['experiment']['severity'])
    eligible = filtered_views(bench, cfg, 'test', False)
    if len(eligible) == 0:
        raise ValueError(f'No Test views available for severity={sev:.2f}')

    # Prefer instance 0 when SupportBench exposes multiple main-severity instances.
    sort_cols = ['view_group_id', 'family']
    if 'instance' in eligible.columns:
        sort_cols.append('instance')
    sort_cols.append('view_id')
    rep = (
        eligible.sort_values(sort_cols)
        .drop_duplicates(['view_group_id', 'family'], keep='first')
        .copy()
    )

    left = pair[['pair_id', 'view_group_id', 'window_start', 'family', 'distribution']].copy()
    cols = ['view_group_id', 'family', 'view_id', 'split', 'window_start', 'severity', 'distribution']
    merged = left.merge(
        rep[cols],
        on=['view_group_id', 'family'],
        how='left',
        suffixes=('_pair', ''),
        validate='one_to_one',
        sort=False,
    )
    if merged.view_id.isna().any():
        miss = merged[merged.view_id.isna()][['pair_id', 'view_group_id', 'family']].head(20)
        raise ValueError(
            f'Paired Test views unavailable at severity={sev:.2f}; first missing pairs:\n{miss.to_string(index=False)}'
        )
    if not np.allclose(merged.severity.astype(float), sev):
        raise AssertionError('Resolved paired evaluation contains wrong severity.')
    if not (merged.window_start_pair.astype(int).to_numpy() == merged.window_start.astype(int).to_numpy()).all():
        raise AssertionError('Paired evaluation changed window_start.')
    if not (merged.distribution_pair.astype(str).to_numpy() == merged.distribution.astype(str).to_numpy()).all():
        raise AssertionError('Paired evaluation changed distribution.')

    out = merged[['view_id', 'view_group_id', 'split', 'window_start', 'family', 'severity', 'distribution']].copy()
    out.insert(0, 'protocol_view_id', pair.pair_id.astype(str).to_numpy())
    return out.reset_index(drop=True)


def materialize_eval_masks(bench, views: pd.DataFrame, path: Path) -> dict:
    M, P, N = len(views), int(bench.P), int(bench.N)
    direct = np.zeros((M, N), dtype=np.uint8)
    future_valid = np.zeros((M, P, N), dtype=np.uint8)
    view_ids = np.empty(M, dtype=f'<U{max(1, max(len(str(v)) for v in views.view_id.astype(str)))}')
    for i, vid in enumerate(views.view_id.astype(str)):
        s = bench.sample(vid)
        direct[i] = s.artificial_mask.any(axis=0).astype(np.uint8)
        future_valid[i] = s.raw_valid_future.astype(np.uint8)
        view_ids[i] = vid
        if (i + 1) % 500 == 0:
            print(f'evaluation masks: {i+1}/{M}', flush=True)
    np.savez_compressed(path, view_id=view_ids, direct_mask=direct, future_valid=future_valid)
    return {
        'rows': M,
        'direct_target_fraction': float(direct.mean()),
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


def _allowed_stages(cfg: dict) -> set[str]:
    vals = cfg.get('protocol', {}).get('allowed_stages', ['development', 'formal', 'evaluation'])
    return {str(x) for x in vals}


def build_stage(bench, cfg: dict, stage: str, force: bool = False) -> Path:
    if stage not in _allowed_stages(cfg):
        raise RuntimeError(
            f"Stage {stage!r} is disabled for {cfg['protocol']['name']}. "
            f"Allowed stages={sorted(_allowed_stages(cfg))}."
        )

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
    paired_info = None

    if stage == 'development':
        train = balanced_unique_group_pick(
            filtered_views(bench, cfg, 'train', False), fams,
            int(cfg['stages']['development']['train_size']), seed,
        )
        val = balanced_unique_group_pick(
            filtered_views(bench, cfg, 'val', False), fams,
            int(cfg['stages']['development']['val_size']), seed + 1,
        )
        split_map = {'train': _canonicalize(train, stage, 'train'), 'val': _canonicalize(val, stage, 'val')}
    elif stage == 'formal':
        dev = root / 'development'
        if not (dev / 'FROZEN.json').exists():
            raise RuntimeError('Development protocol must be frozen before formal protocol.')
        formal_cfg = cfg['stages']['formal']
        exact_balance = bool(cfg.get('protocol', {}).get('formal_train_exact_family_balance', False))
        eligible_train = filtered_views(bench, cfg, 'train', False)
        if exact_balance:
            configured_total = formal_cfg.get('train_size')
            if configured_total in (None, 'all', 0):
                raise ValueError(
                    'Exact-balanced Formal training requires a numeric stages.formal.train_size.'
                )
            train = balanced_unique_group_pick(
                eligible_train, fams, int(configured_total), seed + 10,
            )
        else:
            train = all_unique_group_pick(eligible_train, fams, seed + 10)
        val = pd.read_csv(dev / 'val_views.csv')
        val2 = val.drop(columns=['protocol_view_id']).copy()
        val2.insert(0, 'protocol_view_id', [f'formal_val_{i:08d}' for i in range(len(val2))])
        split_map = {'train': _canonicalize(train, stage, 'train'), 'val': val2}
    elif stage == 'evaluation':
        pair, paired_info = ensure_evaluation_pairing(bench, cfg, force=False)
        test = resolve_paired_evaluation_views(bench, cfg, pair)
        split_map = {'test': test}
    else:
        raise ValueError(stage)

    for split, df in split_map.items():
        df.to_csv(sd / f'{split}_views.csv', index=False)

    speed_name = bench.manifest['primary_channel']
    st = bench.stats['channels'][speed_name]
    source_info = {
        'supportbench_package_version': str(getattr(__import__('supportbench'), '__version__', 'unknown')),
        'supportbench_benchmark_version': str(bench.manifest.get('version', 'unknown')),
        'supportbench_manifest_sha256': canonical_sha256(bench.manifest),
        'supportbench_view_manifest_sha256': sha256_file(bench.sg_dir / 'views' / 'view_manifest.csv'),
    }
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
        'config_sha256': sha256_file(Path(cfg.get('_config_path', Path(cfg['project']['protocol_root']) / 'configs' / 'protocol_s40.yaml'))),
    }

    if stage == 'evaluation':
        pair_csv, pair_frozen = _pairing_paths(cfg)
        summary['paired_evaluation'] = {
            'paired_group_id': str(cfg['protocol']['paired_group_id']),
            'pairing_hash': paired_info['pairing_hash'],
            'pairing_manifest_sha256': sha256_file(pair_csv),
            'pairing_frozen_sha256': sha256_file(pair_frozen),
            'reference_severity': float(paired_info['reference_severity']),
            'access_policy': 'frozen_manifest_direct_access',
        }

    if stage in ('development', 'formal'):
        mask_info = materialize_eval_masks(bench, split_map['val'], sd / 'val_evaluation_masks.npz')
        summary['evaluation_masks'] = {'val': mask_info}
        if stage == 'formal':
            dev_manifest = read_json(root / 'development' / 'stage_manifest.json')
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
        'evaluation_access_locked': False,
    })

    root_manifest = {
        'protocol_name': protocol_name,
        'protocol_version': cfg['version'],
        'dataset': dataset,
        'severity': float(ec['severity']),
        'paired_group_id': str(cfg['protocol']['paired_group_id']),
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
    manifest, views = load_frozen_stage(
        cfg['project']['artifact_root'], dataset, protocol_name, stage,
        require_evaluation_access=False,
    )
    fams = list(cfg['experiment']['families'])
    expected = cfg['stages'][stage]
    for split, df in views.items():
        if df.view_group_id.astype(str).duplicated().any():
            raise AssertionError(f'{stage}/{split}: duplicate view_group_id')
        if set(df.family.astype(str)) - set(fams):
            raise AssertionError(f'{stage}/{split}: unexpected family')
        if not np.allclose(df.severity.astype(float), float(cfg['experiment']['severity'])):
            raise AssertionError(f'{stage}/{split}: wrong severity in frozen views')
        if split in ('train', 'val', 'test'):
            configured = expected.get(f'{split}_size', None)
            if configured not in (None, 'all', 0) and len(df) != int(configured):
                raise AssertionError(f'{stage}/{split}: expected {configured}, got {len(df)}')
        counts = df.family.astype(str).value_counts().to_dict()
        formal_exact = bool(cfg.get('protocol', {}).get('formal_train_exact_family_balance', False))
        require_exact_balance = not (stage == 'formal' and split == 'train') or formal_exact
        if require_exact_balance and max(counts.values()) - min(counts.values()) > 1:
            raise AssertionError(f'{stage}/{split}: family counts not balanced: {counts}')
        if stage == 'formal' and split == 'train':
            missing_families = [f for f in fams if int(counts.get(f, 0)) == 0]
            if missing_families:
                raise AssertionError(f'{stage}/{split}: missing sensing families: {missing_families}')
            if formal_exact:
                total = int(cfg['stages']['formal']['train_size'])
                quotas = {f: total // len(fams) + int(i < total % len(fams)) for i, f in enumerate(fams)}
                bad = {f: (int(counts.get(f, 0)), int(quotas[f])) for f in fams if int(counts.get(f, 0)) != int(quotas[f])}
                if bad:
                    raise AssertionError(f'{stage}/{split}: exact Formal family quotas violated: {bad}')
    if 'train' in views and 'val' in views:
        if set(views['train'].view_id.astype(str)) & set(views['val'].view_id.astype(str)):
            raise AssertionError('Train/Val view_id overlap')
        if set(views['train'].view_group_id.astype(str)) & set(views['val'].view_group_id.astype(str)):
            raise AssertionError('Train/Val view_group_id overlap')
    return manifest


def validate_paired_evaluations(configs: list[dict]) -> dict:
    """Check that every severity uses exactly the same Test base query/family pairs."""
    from .consumer import load_frozen_stage

    if not configs:
        raise ValueError('No configs supplied.')
    rows = []
    base_keys = None
    pair_ids = None
    for cfg in configs:
        manifest, views = load_frozen_stage(
            cfg['project']['artifact_root'], cfg['experiment']['dataset'],
            cfg['protocol']['name'], 'evaluation', require_evaluation_access=False,
        )
        df = views['test'].copy()
        keys = list(zip(df.view_group_id.astype(str), df.family.astype(str), df.window_start.astype(int)))
        ids = df.protocol_view_id.astype(str).tolist()
        if base_keys is None:
            base_keys, pair_ids = keys, ids
        else:
            if keys != base_keys:
                raise AssertionError(f"Paired evaluation mismatch for {cfg['protocol']['name']}")
            if ids != pair_ids:
                raise AssertionError(f"Pair IDs mismatch for {cfg['protocol']['name']}")
        rows.append({
            'protocol_name': cfg['protocol']['name'],
            'severity': float(cfg['experiment']['severity']),
            'stage_hash': manifest['stage_hash'],
            'rows': int(len(df)),
        })
    return {
        'paired_group_id': str(configs[0]['protocol']['paired_group_id']),
        'rows': int(len(base_keys or [])),
        'protocols': rows,
        'status': 'PASS',
    }
