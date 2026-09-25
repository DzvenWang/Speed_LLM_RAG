from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .utils import read_json, stable_seed


def _target_count(total: int, ratio: float) -> int:
    return max(1, min(int(total), int(round(int(total) * float(ratio)))))


def severity_token(severity: float) -> str:
    return f"s{int(round(100*float(severity))):02d}"


@lru_cache(maxsize=64)
def _json_cached(path: str) -> dict:
    return read_json(Path(path), {}) or {}


@lru_cache(maxsize=64)
def _npy_cached(path: str) -> np.ndarray:
    return np.load(path)


FIXED_LAYOUT_FILES = {
    'regional_outage':'regional_layouts.json',
    'directional_outage':'directional_layouts.json',
    'predictive_regional_outage':'predictive_regional_layouts.json',
    'predictive_directional_outage':'predictive_directional_layouts.json',
}

def _layout_catalog(sg_dir: Path, family: str) -> dict:
    if family not in FIXED_LAYOUT_FILES: raise KeyError(f'No fixed layout catalogue for {family}')
    return _json_cached(str((sg_dir/'graph'/'layouts'/FIXED_LAYOUT_FILES[family]).resolve()))


def select_fixed_layout(spec: dict[str, Any], sg_dir: str | Path) -> tuple[np.ndarray, dict]:
    family = str(spec['family'])
    if family not in set(FIXED_LAYOUT_FILES):
        raise ValueError(f'{family} does not use a fixed layout catalogue')
    cat = _layout_catalog(Path(sg_dir), family)
    token = severity_token(float(spec['severity']))
    dist = str(spec.get('distribution','id'))
    layouts = cat.get('severities',{}).get(token,{}).get(dist,[])
    if not layouts:
        raise KeyError(f'No {family} layouts for {token}/{dist} under {sg_dir}')
    idx = int(spec['seed']) % len(layouts)
    item = dict(layouts[idx])
    return np.asarray(item['nodes'], dtype=np.int64), item




def _temporal_outage_shape(ratio: float, H: int, N: int, base_node_fraction: float = 0.75) -> tuple[int, int]:
    """Return (masked_nodes, duration) for temporal outage.

    Historical severities at or below ``base_node_fraction`` retain the exact
    old construction: a fixed fraction of nodes is affected for a duration
    chosen from the requested missing ratio.  For extreme severities above the
    base node fraction (e.g. 90%), duration is already saturated at the full
    history window, so spatial coverage is expanded instead.

    This keeps 20/40/60 behaviour backward-compatible while making 90% a real
    ~90% outage rather than silently clipping to ~75%.
    """
    ratio = float(np.clip(ratio, 0.0, 1.0))
    base = float(np.clip(base_node_fraction, 1e-6, 1.0))
    if ratio <= base + 1e-12:
        node_frac = base
        duration = max(1, min(H, int(round((ratio / base) * H))))
    else:
        node_frac = ratio
        duration = H
    return _target_count(N, node_frac), int(duration)

def effective_missing_ratio(spec: dict[str, Any], H: int, N: int, sg_dir: str | Path | None = None) -> float:
    family = str(spec['family']); ratio = float(spec.get('severity',0.0))
    if family == 'complete' or ratio <= 0:
        return 0.0
    if family == 'entry_random':
        return _target_count(H*N,ratio)/(H*N)
    if family == 'sensor_outage':
        return _target_count(N,ratio)/N
    if family == 'temporal_outage':
        node_frac = float(spec.get('temporal_node_fraction',0.75))
        k_nodes, duration = _temporal_outage_shape(ratio, H, N, node_frac)
        return (k_nodes*duration)/(H*N)
    if family in set(FIXED_LAYOUT_FILES):
        if sg_dir is None:
            return _target_count(N,ratio)/N
        ids,_ = select_fixed_layout(spec,sg_dir)
        return len(ids)/N
    raise KeyError(f'Unknown sensing family: {family}')


def enrich_view_spec(spec: dict[str,Any], H: int, N: int, sg_dir: str | Path) -> dict[str,Any]:
    out = dict(spec)
    if str(out['family']) in set(FIXED_LAYOUT_FILES):
        ids,item = select_fixed_layout(out,sg_dir)
        out['layout_id'] = str(item['layout_id'])
        out['masked_node_count'] = int(len(ids))
        if out['family'] in {'regional_outage','predictive_regional_outage'}:
            out['regional_component_count'] = int(item.get('component_count',1))
            out['regional_largest_component_fraction'] = float(item.get('largest_component_fraction',1.0))
    else:
        out['layout_id'] = ''
        out['masked_node_count'] = int(round(N*float(out.get('severity',0.0)))) if out['family']!='entry_random' else -1
    out['effective_missing_ratio'] = float(effective_missing_ratio(out,H,N,sg_dir))
    return out


def materialize_mask(spec: dict[str, Any], H: int, N: int, sg_dir: str | Path, raw_valid_speed: np.ndarray | None = None) -> np.ndarray:
    family = str(spec['family']); ratio = float(spec.get('severity',0.0)); seed = int(spec['seed'])
    rng = np.random.default_rng(seed); mask = np.zeros((H,N),dtype=bool)
    if family == 'complete' or ratio <= 0:
        return mask
    if family == 'entry_random':
        total=H*N; k=_target_count(total,ratio); idx=rng.choice(total,size=k,replace=False); mask.reshape(-1)[idx]=True
    elif family == 'sensor_outage':
        k=_target_count(N,ratio); nodes=np.sort(rng.choice(N,size=k,replace=False)); mask[:,nodes]=True
    elif family == 'temporal_outage':
        node_frac=float(spec.get('temporal_node_fraction',0.75))
        k_nodes,duration=_temporal_outage_shape(ratio,H,N,node_frac)
        nodes=np.sort(rng.choice(N,size=k_nodes,replace=False))
        start=int(rng.integers(0,max(1,H-duration+1))); mask[start:start+duration,nodes]=True
    elif family in set(FIXED_LAYOUT_FILES):
        nodes,item=select_fixed_layout(spec,sg_dir)
        if 'layout_id' in spec and str(spec.get('layout_id','')) and str(spec['layout_id']) != str(item['layout_id']):
            raise RuntimeError(f"Layout mismatch for {spec.get('view_id')}: manifest={spec['layout_id']} materialized={item['layout_id']}")
        mask[:,nodes]=True
    else:
        raise KeyError(f'Unknown sensing family: {family}')
    if raw_valid_speed is not None:
        mask &= np.asarray(raw_valid_speed,dtype=bool)
    return mask


def make_view_spec(subgraph: str, split: str, window_start: int, family: str, severity: float, instance: int, cfg: dict, distribution: str = 'id') -> dict:
    base=int(cfg['sensing']['global_seed'])
    seed=stable_seed(subgraph,split,window_start,family,severity,instance,distribution,base=base)
    token=f"{int(round(100*severity)):02d}" if family!='complete' else '00'
    view_id=f"{subgraph}__{split}__w{int(window_start):09d}__{family}__s{token}__i{instance:02d}__{distribution}"
    return {
        'view_id':view_id,
        'view_group_id':f"{subgraph}__w{int(window_start):09d}",
        'subgraph':subgraph,'split':split,'window_start':int(window_start),'family':family,
        'severity':float(severity if family!='complete' else 0.0),'instance':int(instance),'distribution':distribution,'seed':int(seed),
        'temporal_node_fraction':float(cfg['sensing'].get('temporal_node_fraction',0.75)),
    }
