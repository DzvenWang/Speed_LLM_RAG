from __future__ import annotations
import os, sys, copy, hashlib, json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]

def project_root():
    return Path(os.environ.get('SPEED_LLM_RAG_ROOT', ROOT)).expanduser().resolve()

def load_config(path=None, dataset=None, target=None, seed=None, smoke=False):
    with open(path or ROOT/'FARE'/'configs'/'full.yaml', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    ex = cfg['experiment']
    if dataset: ex['dataset'] = dataset
    if target: ex['target'] = target
    if seed is not None: ex['seed'] = int(seed)
    if ex['dataset'] not in [f'TraffiDent_Net_{i:02d}' for i in [1,2,3,4,6]]:
        raise ValueError('Supported datasets: Net-01,02,03,04,06')
    if ex['target'] not in ('speed','flow'): raise ValueError('target must be speed or flow')
    cfg['project_root'] = str(project_root())
    if smoke:
        cfg['training']['batch_size'] = min(2, cfg['training']['batch_size'])
        cfg['training']['num_workers'] = 0
        cfg['training']['phase_epochs'] = {k: 1 for k in cfg['training']['phase_epochs']}
        cfg['experiment']['max_train_rows'] = 8
        cfg['experiment']['max_val_rows'] = 4
        # Use well-separated TRAIN memory windows.  The first 8 train views
        # almost completely overlap the first 16 memory windows, so the
        # normal history+horizon exclusion would otherwise remove every
        # smoke-training retrieval candidate.
        cfg['memory']['max_rows'] = 64
        cfg['memory']['sampling'] = 'spread'
        cfg['retrieval']['coarse_topk'] = 8
        cfg['retrieval']['topk'] = 3
    return cfg

def insert_paths():
    root = project_root()
    for p in (root/'Backbone', root/'Base_SupportBench'):
        if p.exists() and str(p) not in sys.path: sys.path.insert(0,str(p))

def cache_path(cfg, stage='formal', severity=None):
    """Resolve dataset-specific frozen Base_SupportBench contract, not Net-06 backbone.yaml."""
    insert_paths()
    root=project_root(); ex=cfg['experiment']; target=ex['target']; dataset=ex['dataset']
    s = f'_s{int(severity):02d}' if stage=='evaluation' else ''
    resolved=root/'Base_SupportBench_artifacts'/'resolved_configs'/target/dataset/f'cache_{stage}{s}.yaml'
    if not resolved.exists(): raise FileNotFoundError(f'Missing resolved SupportBench config: {resolved}. Build that dataset first.')
    from experiment_common.cache import cache_dir, validate_cache
    contract=yaml.safe_load(os.path.expandvars(resolved.read_text(encoding='utf-8').replace('${SPEED_LLM_RAG_ROOT}',str(root)).replace('${TRAFFIDENT_TARGET}',target)))
    if contract['experiment'].get('subgraph',dataset) != dataset: raise RuntimeError('Dataset contract mismatch')
    if contract['experiment'].get('target',target) != target: raise RuntimeError('Target contract mismatch')
    from experiment_common.cache import cache_dir
    out=Path(cache_dir(contract,stage))
    if not out.exists(): raise FileNotFoundError(f'Cache missing: {out}')
    if not (out/'READY.json').exists() or not (out/'cache_manifest.json').exists():
        raise RuntimeError(f'Unready cache: {out}')
    return out

def backbone_checkpoint(cfg, explicit=None):
    if explicit:
        p=Path(explicit).expanduser().resolve()
        if not p.is_file(): raise FileNotFoundError(p)
        return p
    root=project_root(); ex=cfg['experiment']
    p=root/'Backbone_exps'/ex['target']/'STLAformer'/ex['dataset']/'formal'
    pointer=p/'LATEST_CHECKPOINT.txt'
    if pointer.exists():
        t=Path(pointer.read_text(encoding='utf-8').strip()); t=t if t.is_absolute() else root/t
        if t.is_file(): return t.resolve()
    cands=sorted(p.glob(f'*_seed{cfg["experiment"]["seed"]}/best.pt'),key=lambda z:z.stat().st_mtime,reverse=True)
    if not cands: raise FileNotFoundError(f'No pretrained STLAformer checkpoint: {p}')
    return cands[0].resolve()

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def json_write(path, obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf-8')


def config_from_checkpoint(checkpoint, *, smoke=False):
    """Recover the exact architecture/training config embedded in a FARE checkpoint.

    No implicit fallback to today's YAML defaults: architecture configs must
    match saved weights, otherwise evaluation can silently differ from training.
    """
    if 'config' not in checkpoint:
        raise RuntimeError('FARE checkpoint missing saved configuration')
    if not str(checkpoint.get('version','')).startswith('2.1.'):
        raise RuntimeError('FARE v2.1 does not support older model checkpoints')
    cfg=copy.deepcopy(checkpoint['config'])
    required=('experiment','training','model','backbone','graph','retrieval','memory','innovation','evaluation')
    missing=[k for k in required if k not in cfg]
    if missing:raise RuntimeError(f'Checkpoint missing FARE configuration fields: {missing}')
    cfg['project_root']=str(project_root())
    if smoke and cfg['experiment'].get('max_train_rows') is None:
        raise ValueError('--smoke only applies to checkpoints originally trained in smoke mode')
    return cfg
