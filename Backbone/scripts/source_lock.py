#!/usr/bin/env python3
"""Verify bundled vendor source without requiring its .git directory.

Normal verification warns about source-tree SHA drift but does not prevent
training.  --strict-hash is for source restoration / integrity workflows.
The legacy SOURCE_LOCK.json format (PatchSTG only) remains supported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backbone.utils import load_yaml

MODELS = ('STAEformer', 'HimNet', 'BigST', 'STDN', 'PatchSTG', 'STLAformer')
# Files actually loaded by Backbone's vendor adapters: missing files are fatal,
# while a digest drift of otherwise present sources is a warning by default.
REQUIRED_FILES = {
    'STAEformer': ('model/STAEformer.py',),
    'HimNet': ('model/HimNet.py',),
    'BigST': ('model.py',),
    'STDN': ('model.py', 'utils.py'),
    'PatchSTG': ('models/model.py', 'lib/utils.py'),
    'STLAformer': ('model/STLformer.py',),
}
IGNORE_DIRS = {'.git', '__pycache__', '.pytest_cache', '.mypy_cache'}
IGNORE_SUFFIXES = {'.pyc', '.pyo'}


def tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    files = []
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in IGNORE_DIRS for part in rel.parts) or p.suffix in IGNORE_SUFFIXES:
            continue
        files.append(rel)
    for rel in sorted(files, key=lambda x: x.as_posix()):
        h.update(rel.as_posix().encode())
        h.update(b'\0')
        with (root / rel).open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        h.update(b'\0')
    return h.hexdigest()


def source_record(lock_path: Path, model: str):
    """Read the existing per-model lock entry; malformed locks remain fatal."""
    if not lock_path.is_file():
        if model == 'PatchSTG':
            raise SystemExit(f'FAIL SOURCE_LOCK_MISSING {lock_path}')
        print(f'WARNING SOURCE_LOCK_MISSING model={model} path={lock_path}', flush=True)
        return None
    try:
        data = json.loads(lock_path.read_text(encoding='utf-8'))
        records = data['repositories']
        if not isinstance(records, dict):
            raise ValueError('repositories must be an object')
    except (ValueError, KeyError, TypeError) as e:
        raise SystemExit(f'FAIL SOURCE_LOCK_INVALID {lock_path}: {e}') from e
    if model not in records:
        if model == 'PatchSTG':
            raise SystemExit(f'FAIL SOURCE_LOCK_ENTRY_MISSING model={model} path={lock_path}')
        print(f'WARNING SOURCE_LOCK_ENTRY_MISSING model={model} (source available but not hash-locked)', flush=True)
        return None
    rec = records[model]
    if not isinstance(rec, dict) or not isinstance(rec.get('tree_sha256'), str) or not rec['tree_sha256']:
        raise SystemExit(f'FAIL SOURCE_LOCK_ENTRY_INVALID model={model} path={lock_path}')
    return rec


def verify(model: str, cfg: dict, strict_hash: bool = False) -> None:
    repo = ROOT / 'vendor' / model
    if not repo.is_dir():
        raise SystemExit(f'FAIL SOURCE_MISSING model={model} path={repo}')

    for relative in REQUIRED_FILES[model]:
        required = repo / relative
        if not required.is_file():
            raise SystemExit(f'FAIL SOURCE_FILE_MISSING model={model} path={required}')

    lock_path = ROOT / 'vendor' / 'SOURCE_LOCK.json'
    rec = source_record(lock_path, model)
    if rec is None:
        if strict_hash:
            raise SystemExit(f'FAIL SOURCE_LOCK_NOT_AVAILABLE_STRICT model={model}')
        print(f'PASS source_present model={model} hash_status=UNVERIFIED', flush=True)
        return

    # The official_source entry in backbone.yaml describes PatchSTG only.
    # Never compare STLAformer or other models to the PatchSTG identity.
    if model == 'PatchSTG':
        spec = cfg['model']['official_source']
        if (rec.get('commit') != str(spec['commit']) or
                str(rec.get('remote', '')).rstrip('/') != str(spec['remote']).rstrip('/')):
            raise SystemExit(
                f'FAIL SOURCE_IDENTITY_MISMATCH model={model} '
                f'expected_commit={spec["commit"]} lock_commit={rec.get("commit")} '
                f'expected_remote={spec["remote"]} lock_remote={rec.get("remote")}'
            )

    actual = tree_hash(repo)
    expected = rec['tree_sha256']
    if actual != expected:
        print(
            f'WARNING SOURCE_TREE_HASH_MISMATCH model={model} '
            f'expected_tree={expected} actual_tree={actual}',
            flush=True,
        )
        if strict_hash:
            raise SystemExit(f'FAIL SOURCE_TREE_HASH_MISMATCH_STRICT model={model}')
        print(f'PASS source_present model={model} hash_status=WARNING', flush=True)
    else:
        print(f'PASS source model={model} commit={rec.get("commit", "unknown")} tree_sha256={actual}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['verify', 'create'])
    ap.add_argument('--config', default=str(ROOT / 'configs/backbone.yaml'))
    ap.add_argument('--model', choices=MODELS, default='PatchSTG', help='Vendor to verify; legacy default: PatchSTG')
    ap.add_argument('--strict-hash', action='store_true', help='Fail if source tree SHA256 differs from the recorded lock')
    a = ap.parse_args()
    cfg = load_yaml(a.config)
    repo = ROOT / 'vendor' / a.model
    lock_path = ROOT / 'vendor' / 'SOURCE_LOCK.json'
    if a.action == 'create':
        # Preserve the historical PatchSTG-only create contract to avoid
        # overwriting other models' provenance without a known commit/remote.
        if a.model != 'PatchSTG':
            raise SystemExit('FAIL CREATE_REQUIRES_PATCHSTG; use the official-source setup workflow for other models')
        if not repo.is_dir():
            raise SystemExit(f'FAIL SOURCE_MISSING model={a.model} path={repo}')
        spec = cfg['model']['official_source']
        digest = tree_hash(repo)
        rec = {'format': 2, 'repositories': {'PatchSTG': {
            'commit': str(spec['commit']),
            'remote': str(spec['remote']),
            'tree_sha256': digest,
        }}}
        # Keep a previously recorded lock entry for other models if present.
        if lock_path.is_file():
            try:
                previous = json.loads(lock_path.read_text(encoding='utf-8'))
                for name, data in previous.get('repositories', {}).items():
                    if name != 'PatchSTG':
                        rec['repositories'][name] = data
            except (ValueError, AttributeError, TypeError) as e:
                raise SystemExit(f'FAIL SOURCE_LOCK_INVALID {lock_path}: {e}') from e
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps(rec, indent=2, sort_keys=True), encoding='utf-8')
        print('LOCK PatchSTG', digest, flush=True)
        return
    verify(a.model, cfg, strict_hash=a.strict_hash)


if __name__ == '__main__':
    main()
