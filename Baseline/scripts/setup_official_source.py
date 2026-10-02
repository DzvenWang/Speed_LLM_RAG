#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, re, shutil, subprocess
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = ROOT / 'configs' / 'official_sources.yaml'
VENDOR = ROOT / 'vendor'
LOCK_PATH = VENDOR / 'SOURCE_LOCK.json'
SHA_RE = re.compile(r'^[0-9a-f]{40}$')
IGNORE_DIRS = {'.git', '__pycache__', '.pytest_cache', '.mypy_cache'}
IGNORE_SUFFIXES = {'.pyc', '.pyo'}

def run(*args, cwd=None):
    print('RUN', *map(str,args), flush=True)
    subprocess.check_call([str(x) for x in args], cwd=cwd)

def out(*args, cwd=None):
    return subprocess.check_output([str(x) for x in args], cwd=cwd, text=True).strip()

def tree_hash(root: Path) -> str:
    h=hashlib.sha256(); files=[]
    for p in root.rglob('*'):
        if not p.is_file(): continue
        rel=p.relative_to(root)
        if any(x in IGNORE_DIRS for x in rel.parts) or p.suffix in IGNORE_SUFFIXES: continue
        files.append(rel)
    for rel in sorted(files,key=lambda x:x.as_posix()):
        p=root/rel; h.update(rel.as_posix().encode()); h.update(b'\0')
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
        h.update(b'\0')
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description='One-time bootstrap of one pinned official Baseline source.')
    ap.add_argument('--name', required=True)
    ap.add_argument('--force-reclone', action='store_true')
    a=ap.parse_args()
    sources=yaml.safe_load(SOURCES_PATH.read_text())['official_sources']
    if a.name not in sources: raise SystemExit(f'Unknown official source: {a.name}')
    spec=sources[a.name]; repo=VENDOR/a.name; VENDOR.mkdir(parents=True,exist_ok=True)
    if repo.exists() and a.force_reclone:
        print('REMOVE',repo); shutil.rmtree(repo)
    if not repo.exists(): run('git','clone',spec['remote'],repo)
    remote=out('git','remote','get-url','origin',cwd=repo).rstrip('/')
    if remote != str(spec['remote']).rstrip('/'): raise SystemExit(f'{a.name}: remote mismatch: {remote}')
    run('git','fetch','--all','--tags','--prune',cwd=repo)
    if spec.get('policy','exact') == 'exact':
        commit=str(spec['commit']); run('git','checkout','--detach',commit,cwd=repo)
    else:
        ref=str(spec.get('ref','main')); run('git','checkout',ref,cwd=repo); run('git','pull','--ff-only',cwd=repo)
        commit=out('git','rev-parse','HEAD',cwd=repo)
    commit=out('git','rev-parse','HEAD',cwd=repo)
    if not SHA_RE.match(commit): raise SystemExit(f'{a.name}: invalid resolved commit {commit}')
    lock={'format':3,'repositories':{}}
    if LOCK_PATH.exists(): lock=json.loads(LOCK_PATH.read_text())
    lock['format']=3; lock.setdefault('repositories',{})[a.name]={
        'commit':commit,'remote':remote,'tree_sha256':tree_hash(repo),
        'policy':spec.get('policy','exact'),'configured_ref':spec.get('ref')}
    LOCK_PATH.write_text(json.dumps(lock,indent=2,sort_keys=True))
    print('LOCKED',a.name,commit,lock['repositories'][a.name]['tree_sha256'])
    run(str(Path(__import__('sys').executable)), ROOT/'scripts'/'source_lock.py','verify-one',
        '--root',ROOT,'--sources',SOURCES_PATH,'--name',a.name)
    print('OFFICIAL SOURCE READY',a.name)
if __name__=='__main__': main()
