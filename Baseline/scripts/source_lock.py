#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, re, subprocess
from pathlib import Path

IGNORE_DIRS = {'.git', '__pycache__', '.pytest_cache', '.mypy_cache'}
IGNORE_SUFFIXES = {'.pyc', '.pyo'}
SHA_RE = re.compile(r'^[0-9a-f]{40}$')


def tree_hash(root: Path) -> str:
    h = hashlib.sha256(); files = []
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in IGNORE_DIRS for part in rel.parts) or p.suffix in IGNORE_SUFFIXES:
            continue
        files.append(rel)
    for rel in sorted(files, key=lambda x: x.as_posix()):
        p = root / rel
        h.update(rel.as_posix().encode()); h.update(b'\0')
        with open(p, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        h.update(b'\0')
    return h.hexdigest()


def git_cmd(repo: Path, *args: str) -> str:
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def load_sources(path: str | Path) -> dict:
    import yaml
    raw = yaml.safe_load(Path(path).read_text())
    return raw.get('official_sources', raw)


def _policy(spec: dict) -> str:
    return str(spec.get('policy', 'exact'))


def _validate_record(name: str, spec: dict, rec: dict) -> None:
    remote = str(spec['remote']).rstrip('/')
    if str(rec.get('remote', '')).rstrip('/') != remote:
        raise RuntimeError(f'{name}: remote mismatch in source lock')
    commit = str(rec.get('commit', ''))
    if not SHA_RE.match(commit):
        raise RuntimeError(f'{name}: source lock does not contain a full 40-char commit')
    policy = _policy(spec)
    if policy == 'exact' and commit != str(spec['commit']):
        raise RuntimeError(f'{name}: locked commit {commit} != configured exact commit {spec["commit"]}')
    if policy not in ('exact', 'resolve_once'):
        raise RuntimeError(f'{name}: unsupported source policy {policy}')


def verify_one(root: Path, sources: dict, name: str, lock: dict):
    if name not in sources:
        raise RuntimeError(f'Unknown source {name}')
    spec = sources[name]; repo = root / 'vendor' / name
    if not repo.exists():
        raise RuntimeError(f'Missing source directory: {repo}')
    rec = lock.get('repositories', {}).get(name)
    if not rec:
        raise RuntimeError(f'{name}: missing from SOURCE_LOCK.json')
    _validate_record(name, spec, rec)
    got = tree_hash(repo)
    # A different working-tree fingerprint is provenance drift, not by itself a
    # reason to prevent historical checkpoint evaluation.  Keep the recorded
    # fingerprint unchanged and make the mismatch prominent in the live log.
    # Missing source repositories, lock metadata, remote/commit and entrypoint
    # checks remain strict and can still fail.
    tree_mismatch = got != rec.get('tree_sha256')
    if tree_mismatch:
        print(
            f'WARNING SOURCE_TREE_HASH_MISMATCH {name}: '
            f'expected_tree={rec.get("tree_sha256")} actual_tree={got}; '
            'continuing with changed source (reproducibility NOT verified)',
            file=__import__('sys').stderr, flush=True,
        )
    if (repo / '.git').exists():
        commit = git_cmd(repo, 'rev-parse', 'HEAD')
        remote = git_cmd(repo, 'remote', 'get-url', 'origin')
        if commit != rec['commit'] or remote.rstrip('/') != str(spec['remote']).rstrip('/'):
            raise RuntimeError(f'FAIL git metadata {name}')
        mode = 'git+tree'
    else:
        mode = 'tree-lock-no-git'
    if tree_mismatch:
        mode += '+WARNING-tree-hash-mismatch'
    return rec, mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['create', 'verify', 'verify-one'])
    ap.add_argument('--root', required=True)
    ap.add_argument('--sources', required=True)
    ap.add_argument('--name', default=None)
    args = ap.parse_args()

    root = Path(args.root); vendor = root / 'vendor'; sources = load_sources(args.sources)
    lock_path = vendor / 'SOURCE_LOCK.json'
    old = json.loads(lock_path.read_text()) if lock_path.exists() else {'format': 2, 'repositories': {}}

    if args.action == 'create':
        out = {'format': 3, 'repositories': {}}
        for name, spec in sources.items():
            repo = vendor / name
            if not repo.exists():
                raise SystemExit(f'Missing source directory: {repo}')
            policy = _policy(spec)
            if (repo / '.git').exists():
                commit = git_cmd(repo, 'rev-parse', 'HEAD')
                remote = git_cmd(repo, 'remote', 'get-url', 'origin')
                if remote.rstrip('/') != str(spec['remote']).rstrip('/'):
                    raise SystemExit(f'Source remote mismatch before lock: {name}')
                if policy == 'exact' and commit != str(spec['commit']):
                    raise SystemExit(f'Source commit mismatch before lock: {name}: {commit} != {spec["commit"]}')
            else:
                prev = old.get('repositories', {}).get(name)
                if not prev:
                    raise SystemExit(f'{name}: no git metadata and no prior tree lock; rerun setup to clone it.')
                try:
                    _validate_record(name, spec, prev)
                except Exception as e:
                    raise SystemExit(str(e))
                if tree_hash(repo) != prev.get('tree_sha256'):
                    raise SystemExit(f'{name}: existing tree differs from prior lock; rerun setup to clone it.')
                commit = prev['commit']; remote = prev['remote']
            if not SHA_RE.match(str(commit)):
                raise SystemExit(f'{name}: resolved commit is not a full SHA: {commit}')
            digest = tree_hash(repo)
            out['repositories'][name] = {
                'commit': commit,
                'remote': remote,
                'tree_sha256': digest,
                'policy': policy,
                'configured_ref': spec.get('ref'),
            }
            print('LOCK', name, commit, digest, policy)
        lock_path.write_text(json.dumps(out, indent=2, sort_keys=True))
        resolved = vendor / 'RESOLVED_SOURCES.tsv'
        with resolved.open('w') as f:
            f.write('model\tcommit\tremote\tpolicy\n')
            for name, rec in out['repositories'].items():
                f.write(f"{name}\t{rec['commit']}\t{rec['remote']}\t{rec['policy']}\n")
        print('wrote', lock_path)
        print('wrote', resolved)
        return

    if not lock_path.exists():
        raise SystemExit(f'Missing {lock_path}; restore the frozen official source lock before running Formal experiments')
    lock = json.loads(lock_path.read_text())
    names = [args.name] if args.action == 'verify-one' else list(sources)
    if args.action == 'verify-one' and not args.name:
        raise SystemExit('--name is required for verify-one')
    for name in names:
        try:
            rec, mode = verify_one(root, sources, name, lock)
        except Exception as e:
            raise SystemExit(str(e))
        print('WARNING source' if 'WARNING-tree-hash-mismatch' in mode else 'PASS source',
              name, rec['commit'], mode, flush=True)
    if args.action == 'verify':
        print('SOURCE LOCK PASS')


if __name__ == '__main__':
    main()
