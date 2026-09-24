from pathlib import Path
from .utils import stamp


def artifact_dir(cfg, stage):
    return Path(cfg['project']['exp_root']) / '_v12_artifacts' / cfg['experiment']['subgraph'] / stage


def run_dir(cfg, name='DualQueryRetrievalConditionedForecasting', stage='development'):
    return Path(cfg['project']['exp_root']) / name / cfg['experiment']['subgraph'] / stage / f'{stamp()}_seed{cfg["experiment"]["seed"]}'
