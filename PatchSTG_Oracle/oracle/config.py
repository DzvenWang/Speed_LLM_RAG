from __future__ import annotations
import os
from pathlib import Path

from .utils import load_yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def workspace_root(root: Path | None = None) -> Path:
    override=os.environ.get('PATCHSTG_ORACLE_WORKSPACE_ROOT','').strip()
    if override: return Path(override).expanduser().resolve()
    return (root or project_root()).resolve().parent


def load_config(root: Path | None = None):
    """Load the experiment config and bind paths to the current checkout.

    The checked-in YAML retains the Gadi paths as documentation, while Python
    entry points use the repository containing this file.  An explicit
    PATCHSTG_ORACLE_WORKSPACE_ROOT can select another local checkout.
    """
    root=(root or project_root()).resolve(); ws=workspace_root(root)
    cfg=load_yaml(root/'configs'/'oracle.yaml')
    cfg.setdefault('project',{}); cfg['project'].update({
        'root':str(ws),'oracle_root':str(root),
        'exp_root':str(ws/'PatchSTG_Oracle_exps'),'log_root':str(ws/'PatchSTG_Oracle_logs'),
        'cache_root':str(ws/'ExperimentCache'),'backbone_root':str(ws/'Backbone'),
        'backbone_exp_root':str(ws/'Backbone_exps'),'common_root':str(ws/'ExperimentCommon'),
        'supportbench_root':str(ws/'TraffiDent_SupportBench'),
        'benchmark_root':str(ws/'data'/'processed'/'traffident'/'supportbench'),
    })
    cfg.setdefault('shared_contract',{})
    cfg['shared_contract']['config']=str(ws/'ExperimentCommon'/'configs'/'experiment_contract.yaml')
    # Windows spawn can duplicate large NumPy memmaps in DataLoader workers.
    # Direct loading is reliable and fast enough for this 12-step cache.
    if os.name=='nt':
        cfg.setdefault('runtime',{})['num_workers']=0
        cfg['runtime']['persistent_workers']=False
        cfg.setdefault('observation_aware',{})['num_workers']=0
    cfg['local_runtime']={'workspace_root':str(ws),'project_root':str(root)}
    return cfg
