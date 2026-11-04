from __future__ import annotations
import hashlib, json, os, copy
from pathlib import Path
import yaml


def root() -> Path:
    env = os.environ.get('SPEED_LLM_RAG_ROOT')
    return Path(env).expanduser().resolve() if env else Path(__file__).resolve().parents[3]


def load_config(path, *, target=None, dataset=None, seed=None):
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    cfg = copy.deepcopy(cfg)
    ex = cfg['experiment']
    if target: ex['target'] = str(target)
    if dataset: ex['dataset'] = str(dataset)
    if seed is not None: ex['seed'] = int(seed)
    assert ex['target'] in ('speed','flow')
    assert int(ex['history']) == 12 and int(ex['horizon']) == 12, 'STLA source checkpoint expects 12+12'
    return cfg


def json_write(path, x):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x, indent=2, sort_keys=True, default=str),encoding='utf-8')
    tmp.replace(p)


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()


def make_paths(cfg, target, dataset):
    repo=root()
    import sys
    for p in [repo/'Backbone',repo/'Base_SupportBench',repo/'PCFT-STLA']:
        if str(p) not in sys.path:sys.path.insert(0,str(p))
    from backbone.utils import load_yaml
    from backbone.shared_cache import cache_path,validate_shared_cache
    import re
    raw=load_yaml(repo/'Backbone/configs/backbone.yaml')
    def replace(x):
        if isinstance(x,dict):return {k:replace(v) for k,v in x.items()}
        if isinstance(x,list):return [replace(v) for v in x]
        if isinstance(x,str):return re.sub(r'TraffiDent_Net_\d{2}',dataset,x)
        return x
    bcfg=replace(raw)
    # The Backbone shared config target is derived from the environment.
    os.environ['TRAFFIDENT_TARGET']=target
    for stage in ('formal','evaluation'):
        # Evaluations are externally locked/frozen by the common protocol.
        if stage=='formal':validate_shared_cache(bcfg,stage)
    formal=Path(cache_path(bcfg,'formal'))
    eval_paths={}
    for s in cfg['experiment']['severities']:
        validate_shared_cache(bcfg,'evaluation',s)
        eval_paths[int(s)]=Path(cache_path(bcfg,'evaluation',s))
    return repo,formal,eval_paths


def resolve_checkpoint(repo,cfg,explicit):
    target=cfg['experiment']['target']; dataset=cfg['experiment']['dataset']; seed=cfg['experiment']['seed']
    mode=str(cfg['backbone']['mode'])
    base=repo/'Backbone_exps'/target/'STLAformer'/dataset/('formal_complete' if mode=='complete' else 'formal')
    if explicit:
        p=Path(explicit).expanduser();p=(p if p.is_absolute() else repo/p).resolve()
        if not p.is_relative_to(base.resolve()) or p.name!='best.pt':
            raise ValueError(f'Checkpoint must be original Backbone best.pt within {base}, got {p}')
    else:
        candidates=list(base.glob(f'*_seed{seed}/best.pt'))
        if not candidates:raise FileNotFoundError(f'No matching STLAformer checkpoint under {base}')
        p=max(candidates,key=lambda z:z.stat().st_mtime)
    if f'_seed{seed}' not in p.parent.name:
        raise ValueError(f'STLA checkpoint does not match requested seed={seed}: {p}')
    if not p.is_file():raise FileNotFoundError(p)
    return p
