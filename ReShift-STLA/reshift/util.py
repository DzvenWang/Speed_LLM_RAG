"""Side-effect-free helpers for ReShift."""
from __future__ import annotations
import hashlib, json, os, random, tempfile
from pathlib import Path
import numpy as np
import torch


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def source_hash(root):
    root=Path(root)
    files=list((root/'reshift').rglob('*.py'))+list((root/'scripts/reshift').rglob('*.py'))
    files += [root/'corsta'/name for name in ('stla.py','student.py','data.py','observability.py')]
    h=hashlib.sha256()
    for p in sorted(files):h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def atomic_json(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.partial',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def atomic_torch(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.partial',dir=path.parent)
    os.close(fd)
    try:torch.save(obj,tmp);os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'torch':torch.get_rng_state(),'numpy':np.random.get_state(),'random':random.getstate(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(s):
    torch.set_rng_state(s['torch']);np.random.set_state(s['numpy']);random.setstate(s['random'])
    if s.get('cuda') is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(s['cuda'])


def to_device(batch,device):
    return {k:v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}


def masked_mean(x,valid):
    w=valid.to(x.dtype)
    if w.shape!=x.shape:raise ValueError(f'valid shape {w.shape} != value shape {x.shape}')
    return torch.where(w.bool(),x,torch.zeros_like(x)).sum()/w.sum().clamp_min(1)


def checkpoint_load(path):
    return torch.load(path,map_location='cpu',weights_only=False)
