"""Configuration, shared-cache and checkpoint resolution for ReShift-STLA v0.2."""
from __future__ import annotations
import copy, hashlib, os, re, sys
from pathlib import Path
import yaml

_VAR = re.compile(r'\$\{([^}]+)\}')
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT / 'configs' / 'reshift2' / 'reshift2.yaml'
ABLATIONS = ('full', 'base_only', 'no_evolution', 'no_replay', 'no_cf', 'fusion_sum',
             'fusion_learned', 'no_alignment', 'no_risk')


def _expand(v):
    if isinstance(v, dict): return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list): return [_expand(x) for x in v]
    if isinstance(v, str): return _VAR.sub(lambda m: os.environ.get(m.group(1), m.group(0)), v)
    return v


def project_root() -> Path:
    env = os.environ.get('SPEED_LLM_RAG_ROOT')
    if env: return Path(env).expanduser().resolve()
    return PACKAGE_ROOT.parent


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg = _expand(cfg)
    root = project_root(); cfg['project']['root'] = str(root)
    target = os.environ.get('TRAFFIDENT_TARGET', str(cfg['experiment'].get('target', 'speed')))
    if target.startswith('${'): target = 'speed'
    if target not in ('speed', 'flow'): raise RuntimeError(f'TRAFFIDENT_TARGET must be speed/flow, got {target!r}')
    cfg['experiment']['target'] = target
    for k, v in list(cfg['project'].items()):
        if isinstance(v, str):
            cfg['project'][k] = v.replace('${SPEED_LLM_RAG_ROOT}', str(root)).replace('${TRAFFIDENT_TARGET}', target)
    mode = os.environ.get('BACKBONE_MODE', 'incomplete').strip().lower()
    if mode not in ('incomplete', 'complete'):
        raise ValueError(f'BACKBONE_MODE must be incomplete/complete, got {mode!r}')
    cfg['backbone']['training_mode'] = mode
    cfg['experiment']['backbone_mode'] = mode
    for key in ('exp_root', 'log_root', 'store_root'):
        cfg['project'][key] = str(Path(cfg['project'][key]) / mode)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    ab = cfg['training'].get('ablation', 'full')
    if ab not in ABLATIONS: raise ValueError(f'Unknown ablation {ab!r}; choose from {ABLATIONS}')
    if cfg['fusion']['mode'] not in ('coverage', 'sum', 'learned'):
        raise ValueError('fusion.mode must be coverage|sum|learned')
    if int(cfg['replay']['top_r']) > 2 * int(cfg['retrieval']['top_k']):
        raise ValueError('replay.top_r cannot exceed the number of retrieved candidates (2*top_k)')
    if cfg['counterfactual']['target'] not in ('frozen_teacher',):
        raise ValueError('counterfactual.target must be frozen_teacher')


def apply_ablation(cfg: dict, ablation: str) -> dict:
    """Return a copy whose branch switches implement an independently trained ablation."""
    cfg = copy.deepcopy(cfg); cfg['training']['ablation'] = ablation
    br = cfg['model']['branches'] = dict(cfg['model'].get('branches', {}))
    for k in ('evolution', 'replay', 'cf'): br.setdefault(k, True)
    if ablation == 'base_only': br.update(evolution=False, replay=False, cf=False)
    if ablation == 'no_evolution': br['evolution'] = False
    if ablation == 'no_replay': br['replay'] = False
    if ablation == 'no_cf': br['cf'] = False
    if ablation == 'fusion_sum': cfg['fusion']['mode'] = 'sum'
    if ablation == 'fusion_learned': cfg['fusion']['mode'] = 'learned'
    if ablation == 'no_alignment': cfg['training']['alignment_weight'] = 0.0
    if ablation == 'no_risk': cfg['model']['use_risk_gate'] = False
    validate_config(cfg)
    return cfg


def add_repo_paths(cfg: dict) -> None:
    for p in (Path(cfg['project']['backbone_root']), Path(cfg['project']['supportbench_root'])):
        if str(p) not in sys.path: sys.path.insert(0, str(p))


def backbone_protocol_config(cfg: dict) -> dict:
    add_repo_paths(cfg)
    from backbone.utils import load_yaml
    bcfg = load_yaml(Path(cfg['project']['backbone_root']) / 'configs' / 'backbone.yaml')
    dataset = str(cfg['experiment']['dataset'])

    def replace_dataset(v):
        if isinstance(v, dict): return {k: replace_dataset(x) for k, x in v.items()}
        if isinstance(v, list): return [replace_dataset(x) for x in v]
        if isinstance(v, str): return re.sub(r'TraffiDent_Net_\d{2}', dataset, v)
        return v
    return replace_dataset(bcfg)


def cache_path(cfg: dict, stage: str = 'formal', severity: int | None = None) -> Path:
    add_repo_paths(cfg)
    from backbone.shared_cache import cache_path as _cache, validate_shared_cache
    bcfg = backbone_protocol_config(cfg)
    validate_shared_cache(bcfg, stage, severity)
    return Path(_cache(bcfg, stage, severity))


def resolve_stla_checkpoint(cfg: dict, explicit: str | None = None) -> Path:
    root = Path(cfg['project']['root'])
    raw = explicit or os.environ.get('BACKBONE_CHECKPOINT')
    if raw:
        p = Path(raw).expanduser(); p = p if p.is_absolute() else root / p
        if not p.is_file(): raise FileNotFoundError(p)
        return p.resolve()
    formal = root / 'Backbone_exps' / cfg['experiment']['target'] / 'STLAformer' / cfg['experiment']['dataset'] / 'formal'
    pointer = formal / 'LATEST_CHECKPOINT.txt'
    if pointer.is_file():
        p = Path(pointer.read_text(encoding='utf-8').strip()); p = p if p.is_absolute() else root / p
        if p.is_file(): return p.resolve()
    cands = sorted(formal.glob(f'*_seed{int(cfg["backbone"]["reference_seed"])}/best.pt'),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands: raise FileNotFoundError(f'No STLAformer checkpoint under {formal}; set BACKBONE_CHECKPOINT')
    return cands[0].resolve()


def relocate_saved_config(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg); original = str(cfg['project']['root']); new = str(project_root())
    if original != new:
        cfg['project'] = {k: (v.replace(original, new) if isinstance(v, str) else v) for k, v in cfg['project'].items()}
        cfg['project']['root'] = new
    return cfg


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()
