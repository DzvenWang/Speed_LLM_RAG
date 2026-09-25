#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backbone.utils import load_yaml
IGNORE_DIRS={'.git','__pycache__','.pytest_cache','.mypy_cache'}; IGNORE_SUFFIXES={'.pyc','.pyo'}

def tree_hash(root: Path) -> str:
    h=hashlib.sha256(); files=[]
    for p in root.rglob('*'):
        if not p.is_file(): continue
        rel=p.relative_to(root)
        if any(part in IGNORE_DIRS for part in rel.parts) or p.suffix in IGNORE_SUFFIXES: continue
        files.append(rel)
    for rel in sorted(files,key=lambda x:x.as_posix()):
        p=root/rel; h.update(rel.as_posix().encode()); h.update(b'\0')
        with open(p,'rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
        h.update(b'\0')
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['verify','create']); ap.add_argument('--config',default=str(ROOT/'configs/backbone.yaml')); a=ap.parse_args()
    cfg=load_yaml(a.config); spec=cfg['model']['official_source']; repo=ROOT/'vendor'/'PatchSTG'; lock_path=ROOT/'vendor'/'SOURCE_LOCK.json'
    if not repo.exists(): raise SystemExit(f'Missing {repo}')
    if a.action=='create':
        rec={'format':2,'repositories':{'PatchSTG':{'commit':str(spec['commit']),'remote':str(spec['remote']),'tree_sha256':tree_hash(repo)}}}; lock_path.write_text(json.dumps(rec,indent=2,sort_keys=True)); print('LOCK PatchSTG',rec['repositories']['PatchSTG']['tree_sha256']); return
    if not lock_path.exists(): raise SystemExit(f'Missing {lock_path}')
    rec=json.loads(lock_path.read_text())['repositories']['PatchSTG']; got=tree_hash(repo)
    if rec['commit']!=str(spec['commit']) or rec['remote'].rstrip('/')!=str(spec['remote']).rstrip('/') or rec['tree_sha256']!=got:
        raise SystemExit(f'FAIL PatchSTG source lock expected={rec.get("tree_sha256")} actual={got}')
    print('PASS source PatchSTG',rec['commit'],got)
if __name__=='__main__': main()
