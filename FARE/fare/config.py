"""FARE v4 paths are resolved from the source tree, independent of the launch CWD."""
from __future__ import annotations
from pathlib import Path
import os, json, hashlib, yaml, sys

PACKAGE_ROOT=Path(__file__).resolve().parents[1]

def root():
    return Path(os.environ.get('SPEED_LLM_RAG_ROOT', PACKAGE_ROOT.parent)).expanduser().resolve()

def project_path(path, *, must_exist=False):
    """Resolve CLI paths without requiring the caller to cd into the repo.

    Absolute paths are honored; existing relative paths may be relative to the
    current directory, workspace root, or the FARE package. Missing output
    paths default to the workspace root. No HPC-specific path is hardcoded.
    """
    p=Path(os.path.expandvars(str(path))).expanduser()
    if p.is_absolute():
        result=p.resolve()
    else:
        candidates=[Path.cwd()/p, root()/p, PACKAGE_ROOT/p]
        result=next((x.resolve() for x in candidates if x.exists()),(root()/p).resolve())
    if must_exist and not result.exists():
        raise FileNotFoundError(f'Cannot resolve project path: {path}; looked under current directory, {root()}, and {PACKAGE_ROOT}')
    return result


def read_config(path=None, dataset=None, target=None, seed=None, smoke=False):
    p=project_path(path, must_exist=True) if path else PACKAGE_ROOT/'configs'/'pilot.yaml'
    c=yaml.safe_load(p.read_text(encoding='utf8'))
    c.setdefault('evaluation',{}).setdefault('include_val', False)
    c.setdefault('analysis',{}).setdefault('counterfactual_split','s40')
    if not isinstance(c['evaluation']['include_val'], bool):
        raise ValueError('evaluation.include_val must be boolean')
    e=c['experiment']
    if dataset:e['dataset']=dataset
    if target:e['target']=target
    if seed is not None:e['seed']=int(seed)
    if e['target'] not in ('speed','flow'):raise ValueError('target must be speed or flow')
    if e['dataset'] not in {f'TraffiDent_Net_{i:02}' for i in (1,2,3,4,6)}:raise ValueError('Unsupported TraffiDent network')
    if c['model']['history']!=12 or c['model']['horizon']!=12:raise ValueError('This study protocol uses H=P=12')
    if smoke:
        c['experiment']['train_limit']=min(12,int(e.get('train_limit') or 12))
        c['experiment']['val_limit']=min(8,int(e.get('val_limit') or 8))
        c['experiment']['test_limit']=4
        c['memory']['max_episodes']=min(64,c['memory']['max_episodes'])
        c['training']['batch_size']=2
        c['training']['workers']=0
        c['training']['stage_epochs']={k:1 for k in c['training']['stage_epochs']}
        c['teacher']['max_queries']=min(6,c['teacher']['max_queries'])
        c['analysis']['max_counterfactual_queries']=4
    if c['retrieval']['final_k']>c['retrieval']['shortlist_k'] or c['retrieval']['shortlist_k']>c['retrieval']['coarse_k']:
        raise ValueError('Required: final_k <= shortlist_k <= coarse_k')
    return c

def add_shared_paths():
    for name in ('Base_SupportBench','Backbone'):
        p=root()/name
        if p.exists() and str(p) not in sys.path:sys.path.insert(0,str(p))

def cache_dir(c, split='formal', severity=None):
    """Use actual Base_SupportBench resolved config; no guessed cache path."""
    add_shared_paths()
    e=c['experiment']; suffix=f'_s{severity:02}' if split=='evaluation' else ''
    y=root()/'Base_SupportBench_artifacts'/'resolved_configs'/e['target']/e['dataset']/f'cache_{split}{suffix}.yaml'
    if not y.is_file():raise FileNotFoundError(f'Build Base_SupportBench cache before FARE: {y}')
    cfg=yaml.safe_load(os.path.expandvars(y.read_text().replace('${SPEED_LLM_RAG_ROOT}',str(root())).replace('${TRAFFIDENT_TARGET}',e['target'])))
    from experiment_common.cache import cache_dir as resolve
    out=Path(resolve(cfg,split)).expanduser()
    if not out.is_absolute():out=(root()/out).resolve()
    if not (out/'READY.json').exists() or not (out/'cache_manifest.json').exists():
        raise RuntimeError(f'Missing complete Base_SupportBench cache: {out}')
    return out

def backbone_checkpoint(c, explicit=None):
    if explicit:
        p=project_path(explicit,must_exist=True)
        if not p.is_file():raise FileNotFoundError(p)
        return p
    e=c['experiment'];base=root()/'Backbone_exps'/e['target']/'STLAformer'/e['dataset']/'formal'
    pointer=base/'LATEST_CHECKPOINT.txt'
    if pointer.exists():
        raw=Path(pointer.read_text().strip());candidate=raw if raw.is_absolute() else root()/raw
        if candidate.exists():return candidate.resolve()
    cands=sorted(base.glob(f'*_seed{e["seed"]}/best.pt'),key=lambda p:p.stat().st_mtime,reverse=True)
    if not cands:raise FileNotFoundError(f'Missing STLAformer checkpoint: {base}')
    return cands[0].resolve()

def hash_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def dump_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
