#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path

IGNORE_DIRS={'.git','__pycache__','.pytest_cache','.mypy_cache'}
IGNORE_SUFFIXES={'.pyc','.pyo'}


def tree_hash(root: Path) -> str:
    h=hashlib.sha256(); files=[]
    for p in root.rglob('*'):
        if not p.is_file(): continue
        rel=p.relative_to(root)
        if any(part in IGNORE_DIRS for part in rel.parts): continue
        if p.suffix in IGNORE_SUFFIXES: continue
        files.append(rel)
    for rel in sorted(files,key=lambda x:x.as_posix()):
        p=root/rel; h.update(rel.as_posix().encode()); h.update(b'\0')
        with open(p,'rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
        h.update(b'\0')
    return h.hexdigest()


def git_cmd(repo: Path,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()


def load_sources(path):
    import yaml
    raw=yaml.safe_load(Path(path).read_text())
    return raw.get('official_sources',raw)


def verify_one(root, sources, name, lock):
    if name not in sources: raise RuntimeError(f'Unknown source {name}')
    spec=sources[name]; repo=root/'vendor'/name
    if not repo.exists(): raise RuntimeError(f'Missing source directory: {repo}')
    rec=lock.get('repositories',{}).get(name,{})
    got=tree_hash(repo)
    if rec.get('commit')!=str(spec['commit']) or rec.get('remote','').rstrip('/')!=str(spec['remote']).rstrip('/') or got!=rec.get('tree_sha256'):
        raise RuntimeError(f'FAIL source tree {name}: expected_tree={rec.get("tree_sha256")} actual_tree={got}')
    if (repo/'.git').exists():
        commit=git_cmd(repo,'rev-parse','HEAD'); remote=git_cmd(repo,'remote','get-url','origin')
        if commit!=str(spec['commit']) or remote.rstrip('/')!=str(spec['remote']).rstrip('/'):
            raise RuntimeError(f'FAIL git metadata {name}')
        mode='git+tree'
    else:
        mode='tree-lock-no-git'
    return rec, mode


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['create','verify','verify-one']); ap.add_argument('--root',required=True); ap.add_argument('--sources',required=True); ap.add_argument('--name',default=None); a=ap.parse_args()
    root=Path(a.root); vendor=root/'vendor'; sources=load_sources(a.sources); lock_path=vendor/'SOURCE_LOCK.json'
    old=json.loads(lock_path.read_text()) if lock_path.exists() else {'format':1,'repositories':{}}
    if a.action=='create':
        out={'format':2,'repositories':{}}
        for name,spec in sources.items():
            repo=vendor/name
            if not repo.exists(): raise SystemExit(f'Missing source directory: {repo}')
            if (repo/'.git').exists():
                commit=git_cmd(repo,'rev-parse','HEAD'); remote=git_cmd(repo,'remote','get-url','origin')
                if commit!=str(spec['commit']) or remote.rstrip('/')!=str(spec['remote']).rstrip('/'):
                    raise SystemExit(f'Source mismatch before lock: {name}')
            else:
                prev=old.get('repositories',{}).get(name)
                if not prev or prev.get('commit')!=str(spec['commit']) or prev.get('remote','').rstrip('/')!=str(spec['remote']).rstrip('/'):
                    raise SystemExit(f'{name}: no git metadata and no compatible prior tree lock; rerun setup to clone it.')
                if tree_hash(repo)!=prev.get('tree_sha256'):
                    raise SystemExit(f'{name}: existing tree differs from prior lock; rerun setup to clone it.')
                commit=str(spec['commit']); remote=str(spec['remote'])
            digest=tree_hash(repo)
            out['repositories'][name]={'commit':commit,'remote':remote,'tree_sha256':digest}
            print('LOCK',name,commit,digest)
        lock_path.write_text(json.dumps(out,indent=2,sort_keys=True)); print('wrote',lock_path)
    else:
        if not lock_path.exists(): raise SystemExit(f'Missing {lock_path}; rerun setup_official_models.sh')
        lock=json.loads(lock_path.read_text())
        names=[a.name] if a.action=='verify-one' else list(sources)
        if a.action=='verify-one' and not a.name: raise SystemExit('--name is required for verify-one')
        for name in names:
            try: rec,mode=verify_one(root,sources,name,lock)
            except Exception as e: raise SystemExit(str(e))
            print('PASS source',name,rec['commit'],mode)
        if a.action=='verify': print('SOURCE LOCK PASS')

if __name__=='__main__': main()
