from __future__ import annotations
from pathlib import Path
import hashlib, json, random
import numpy as np
import torch
import yaml


def load_yaml(path):
    with open(path,'r') as f: return yaml.safe_load(f)


def write_json(path,obj):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,sort_keys=True,default=str))


def sha256_file(path, chunk=1024*1024):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(chunk),b''): h.update(b)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False


def log(msg): print(str(msg),flush=True)


def deterministic_split(n, dev_n, final_n, seed):
    rng=np.random.default_rng(int(seed)); p=rng.permutation(int(n))
    dev_n=min(int(dev_n),int(n)); final_n=min(int(final_n),max(int(n)-dev_n,0))
    return p[:dev_n].astype(np.int64),p[dev_n:dev_n+final_n].astype(np.int64)
