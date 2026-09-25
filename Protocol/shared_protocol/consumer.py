from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from .utils import canonical_view_hash


STAGE_SPLITS = {
    'development': ('train', 'val'),
    'formal': ('train', 'val'),
    'evaluation': ('test',),
}


def stage_dir(artifact_root: str | Path, dataset: str, protocol_name: str, stage: str) -> Path:
    return Path(artifact_root) / dataset / protocol_name / stage


def _require_evaluation_access(artifact_root: str | Path, dataset: str, manifest: dict) -> None:
    """Backward-compatible no-op: Protocol v2.0.2 has no Test access lock."""
    return None


def load_frozen_stage(
    artifact_root: str | Path,
    dataset: str,
    protocol_name: str,
    stage: str,
    require_evaluation_access: bool = True,
):
    if stage not in STAGE_SPLITS:
        raise ValueError(f'Unsupported protocol stage: {stage}')
    sd = stage_dir(artifact_root, dataset, protocol_name, stage)
    freeze = sd / 'FROZEN.json'
    manifest_path = sd / 'stage_manifest.json'
    if not freeze.exists() or not manifest_path.exists():
        raise FileNotFoundError(f'Protocol stage is not frozen: {sd}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    frozen = json.loads(freeze.read_text(encoding='utf-8'))
    if frozen.get('stage_hash') != manifest.get('stage_hash'):
        raise RuntimeError(f'Protocol freeze hash mismatch: {sd}')
    # Protocol v2.0.2: no evaluation-access grant file is required.
    # The argument is retained only for compatibility with older callers.
    views = {}
    for split in STAGE_SPLITS[stage]:
        p = sd / f'{split}_views.csv'
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_csv(p)
        expected = manifest['splits'][split]['view_hash']
        got = canonical_view_hash(df)
        if got != expected:
            raise RuntimeError(f'Protocol view hash mismatch for {stage}/{split}: expected={expected} got={got}')
        views[split] = df
    return manifest, views


def protocol_stage_hash(artifact_root: str | Path, dataset: str, protocol_name: str, stage: str) -> str:
    manifest, _ = load_frozen_stage(
        artifact_root, dataset, protocol_name, stage,
        require_evaluation_access=False,
    )
    return str(manifest['stage_hash'])
