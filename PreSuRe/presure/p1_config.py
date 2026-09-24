from __future__ import annotations
from pathlib import Path
import os
from .utils import load_yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def workspace_root(root: Path | None = None) -> Path:
    env = os.environ.get("PRESURE_WORKSPACE_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    root = (root or project_root()).resolve()
    return root.parent


def load_config(root: Path | None = None):
    root = (root or project_root()).resolve()
    cfg = load_yaml(root / "configs" / "presure_p1.yaml")
    ws = workspace_root(root)
    cfg.setdefault("project", {})
    cfg["project"].update({
        "root": str(ws),
        "presure_root": str(root),
        "exp_root": str(ws / "PreSuRe_exps"),
        "log_root": str(ws / "PreSuRe_logs"),
        "cache_root": str(ws / "ExperimentCache"),
        "backbone_root": str(ws / "Backbone"),
        "backbone_exp_root": str(ws / "Backbone_exps"),
        "common_root": str(ws / "ExperimentCommon"),
        "supportbench_root": str(ws / "TraffiDent_SupportBench"),
        "benchmark_root": str(ws / "data" / "processed" / "traffident" / "supportbench"),
    })
    cfg.setdefault("shared_contract", {})
    cfg["shared_contract"]["config"] = str(ws / "ExperimentCommon" / "configs" / "experiment_contract.yaml")
    return cfg
