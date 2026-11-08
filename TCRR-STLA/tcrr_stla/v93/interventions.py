"""Deterministic TRAIN-only multi-failure interventions using official SupportBench.

SupportBench: True means artificial MISSING. TCRR: True means OBSERVED.
Original source invalid positions are never reintroduced.
"""
from __future__ import annotations
import importlib, sys, hashlib, json
from pathlib import Path
import numpy as np
import yaml

FAMILIES=('entry_random','sensor_outage','temporal_outage','regional_outage','directional_outage')
SEVERITIES=(20,40,60,90)


def _base_sensing(project_root):
    base=Path(project_root)/'Base_SupportBench'
    if not (base/'supportbench'/'sensing.py').is_file():
        raise FileNotFoundError(f'Base_SupportBench sensing source not found: {base}')
    if str(base) not in sys.path:sys.path.insert(0,str(base))
    return importlib.import_module('supportbench.sensing')


def benchmark_directory(project_root,target,dataset):
    path=Path(project_root)/'data/processed/traffident/base_supportbench'/target/dataset
    if not (path/'graph'/'layouts').is_dir():
        raise FileNotFoundError(f'Official failure layout directory missing: {path}')
    return path


def _base_config(project_root):
    file=Path(project_root)/'Base_SupportBench/configs/base_supportbench.yaml'
    if not file.is_file():raise FileNotFoundError(file)
    cfg=yaml.safe_load(file.read_text(encoding='utf-8'))
    if 'sensing' not in cfg or 'global_seed' not in cfg['sensing']:
        raise RuntimeError('Official Base_SupportBench sensing.global_seed missing')
    return cfg


def enumerate_intervention(query_index,view,views_per_query):
    """Rotate the 20 family×severity cells; 0 is original official view.

    Across consecutive queries, each additional view covers all 20 cells.
    """
    if view==0:return None
    if not(1<=view<views_per_query):raise ValueError('Bad synthetic view index')
    cell=(int(query_index)*(views_per_query-1)+(view-1))%20
    return FAMILIES[cell//4],SEVERITIES[cell%4]


def observed_intervention(reference_valid,query_index,window_start,view,views_per_query,
                          dataset,project_root,target):
    """Return observed mask, metadata, with 1=True=observed."""
    ref=np.asarray(reference_valid,dtype=bool)
    if ref.ndim!=2:raise ValueError('Expected reference mask [L,N]')
    if view==0:
        return ref.copy(),{'family':'complete_reference','severity':0,'layout_id':'',
                           'view_id':f'{dataset}__train__w{int(window_start):09d}__reference'}
    family,sev=enumerate_intervention(query_index,view,views_per_query)
    sensing=_base_sensing(project_root);sg=benchmark_directory(project_root,target,dataset)
    config=_base_config(project_root)
    # The split identity is TRAIN and the official generator is deterministic.
    spec=sensing.make_view_spec(dataset,'train',int(window_start),family,sev/100.0,
                                int(view),config,distribution='id')
    spec=sensing.enrich_view_spec(spec,ref.shape[0],ref.shape[1],sg)
    artificial=np.asarray(sensing.materialize_mask(spec,*ref.shape,sg),dtype=bool)
    if artificial.shape!=ref.shape:raise RuntimeError('SupportBench intervention shape mismatch')
    observed=ref & ~artificial
    if np.any(observed & ~ref):raise RuntimeError('Intervention exposed originally invalid observations')
    return observed,spec


def plan_digest(project_root,target,dataset,views_per_query):
    """Identity includes code, configuration and concrete official spatial layouts."""
    from ..config import file_sha256
    sg=benchmark_directory(project_root,target,dataset)
    paths=[Path(__file__),Path(project_root)/'Base_SupportBench/supportbench/sensing.py',
           Path(project_root)/'Base_SupportBench/configs/base_supportbench.yaml',
           sg/'graph/layouts/regional_layouts.json',sg/'graph/layouts/directional_layouts.json']
    for p in paths:
        if not p.is_file():raise FileNotFoundError(p)
    payload={'views_per_query':views_per_query,'source_hashes':{str(p):file_sha256(p) for p in paths}}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
