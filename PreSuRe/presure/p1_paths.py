from pathlib import Path


def artifact_dir(cfg, stage):
    return Path(cfg['project']['exp_root']) / '_p1_episode_artifacts' / cfg['experiment']['subgraph'] / stage


def run_root(cfg, stage, run_tag):
    return Path(cfg['project']['exp_root']) / 'PreSuRe_P1' / cfg['experiment']['subgraph'] / stage / str(run_tag)


def variant_dir(cfg, stage, run_tag, variant):
    return run_root(cfg, stage, run_tag) / str(variant)
