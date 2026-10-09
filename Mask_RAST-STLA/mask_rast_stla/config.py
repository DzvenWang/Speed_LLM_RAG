from __future__ import annotations
import os, re, yaml
from pathlib import Path

_VAR = re.compile(r'\$\{([^}]+)\}')

def _expand(v):
    if isinstance(v, dict): return {k:_expand(x) for k,x in v.items()}
    if isinstance(v, list): return [_expand(x) for x in v]
    if isinstance(v, str):
        return _VAR.sub(lambda m: os.environ.get(m.group(1), m.group(0)), v)
    return v

def project_root() -> Path:
    env=os.environ.get('SPEED_LLM_RAG_ROOT')
    if env: return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]

def load_config(path: str|Path|None=None) -> dict:
    if path is None: path=Path(__file__).resolve().parents[1]/'configs'/'mask_rast_stla.yaml'
    with open(path,'r',encoding='utf-8') as f: cfg=yaml.safe_load(f)
    cfg=_expand(cfg)
    root=project_root(); cfg['project']['root']=str(root)
    target=os.environ.get('TRAFFIDENT_TARGET', str(cfg['experiment'].get('target','speed')))
    if target.startswith('${'): target='speed'
    if target not in ('speed','flow'): raise RuntimeError(f'TRAFFIDENT_TARGET must be speed/flow, got {target!r}')
    cfg['experiment']['target']=target
    for k,v in list(cfg['project'].items()):
        if isinstance(v,str): cfg['project'][k]=v.replace('${SPEED_LLM_RAG_ROOT}',str(root)).replace('${TRAFFIDENT_TARGET}',target)
    return cfg

def add_repo_paths(cfg: dict) -> None:
    import sys
    for p in (Path(cfg['project']['backbone_root']), Path(cfg['project']['supportbench_root'])):
        if str(p) not in sys.path: sys.path.insert(0,str(p))

def backbone_protocol_config(cfg: dict) -> dict:
    add_repo_paths(cfg)
    from backbone.utils import load_yaml
    p=Path(cfg['project']['backbone_root'])/'configs'/'backbone.yaml'
    bcfg=load_yaml(p)
    # Inherit the shared Backbone protocol; only change the explicitly chosen
    # dataset. Do not silently replace masking/splits/seed/normalization.
    dataset=str(cfg['experiment']['dataset'])
    def replace_dataset(v):
        if isinstance(v,dict):return {k:replace_dataset(x) for k,x in v.items()}
        if isinstance(v,list):return [replace_dataset(x) for x in v]
        if isinstance(v,str):return re.sub(r'TraffiDent_Net_\d{2}',dataset,v)
        return v
    return replace_dataset(bcfg)

def cache_path(cfg: dict, stage: str='formal', severity: int|None=None) -> Path:
    add_repo_paths(cfg)
    from backbone.shared_cache import cache_path as _cache, validate_shared_cache
    bcfg=backbone_protocol_config(cfg)
    validate_shared_cache(bcfg,stage,severity)
    return Path(_cache(bcfg,stage,severity))

def resolve_stla_checkpoint(cfg: dict, explicit: str|None=None) -> Path:
    root=Path(cfg['project']['root']); env_name=str(cfg['backbone'].get('checkpoint_env',''))
    raw=explicit or os.environ.get('BACKBONE_CHECKPOINT') or (os.environ.get(env_name) if env_name else None)
    if raw:
        p=Path(raw).expanduser(); p=p if p.is_absolute() else root/p
        if not p.is_file(): raise FileNotFoundError(p)
        return p.resolve()
    formal=root/'Backbone_exps'/cfg['experiment']['target']/'STLAformer'/cfg['experiment']['dataset']/'formal'
    pointer=formal/'LATEST_CHECKPOINT.txt'
    if pointer.is_file():
        s=pointer.read_text(encoding='utf-8').strip(); p=Path(s); p=p if p.is_absolute() else root/p
        if p.is_file(): return p.resolve()
    cands=sorted(formal.glob(f'*_seed{int(cfg["backbone"]["reference_seed"])}/best.pt'), key=lambda p:p.stat().st_mtime, reverse=True)
    if not cands: raise FileNotFoundError(f'No STLAformer checkpoint under {formal}. Set {env_name}.')
    return cands[0].resolve()


def relocate_saved_config(cfg: dict) -> dict:
    """Allow an identical saved experiment to evaluate on local PC or Gadi."""
    import copy
    cfg=copy.deepcopy(cfg)
    original=str(cfg['project']['root'])
    new=str(project_root())
    if original!=new:
        cfg['project']={k:(v.replace(original,new) if isinstance(v,str) else v)
                        for k,v in cfg['project'].items()}
        cfg['project']['root']=new
    return cfg

def file_sha256(path):
    import hashlib
    h=hashlib.sha256()
    with open(path,'rb') as fh:
        for chunk in iter(lambda:fh.read(4*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()
