from __future__ import annotations
from pathlib import Path
from datetime import datetime
import hashlib, json, os, random, re
import numpy as np
import torch
import yaml

_TS_RE = re.compile(r'^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]')


def load_yaml(path):
    with open(path,'r',encoding='utf-8') as f:
        return yaml.safe_load(f)


def write_json(path,obj):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,sort_keys=True,default=str),encoding='utf-8')


def sha256_file(path, chunk=1024*1024):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(chunk),b''):
            h.update(b)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def ensure_dir(p):
    p=Path(p); p.mkdir(parents=True,exist_ok=True); return p


def stamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def wall_stamp():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def timestamp_text(msg):
    """Prefix every physical line with a wall-clock timestamp.

    If a line is already timestamped in the canonical format, it is preserved. This
    keeps nested/local pipeline logging readable while guaranteeing that warnings,
    progress lines, and ordinary prints can all be normalized by the parent runner.
    """
    text=str(msg)
    lines=text.splitlines() or ['']
    out=[]
    for line in lines:
        if _TS_RE.match(line):
            out.append(line)
        else:
            out.append(f'[{wall_stamp()}] {line}')
    return '\n'.join(out)


def log(msg):
    print(timestamp_text(msg),flush=True)


def atomic_npy(path, arr):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+'.tmp')
    with open(tmp,'wb') as f:
        np.save(f,arr)
    os.replace(tmp,p)
