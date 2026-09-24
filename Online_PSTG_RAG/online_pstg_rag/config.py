from __future__ import annotations
from pathlib import Path
import os
from .utils import load_yaml


def load_config(project_dir: Path, project_root_override=None):
    cfg = load_yaml(project_dir / "configs" / "online_pstg_rag.yaml")
    if project_root_override:
        root = Path(project_root_override).expanduser().resolve()
    elif os.environ.get("SPEED_LLM_RAG_ROOT"):
        root = Path(os.environ["SPEED_LLM_RAG_ROOT"]).expanduser().resolve()
    else:
        root = project_dir.resolve().parent
    p = cfg["paths"]
    cfg["project"] = {
        "root": str(root),
        "rag_root": str(project_dir.resolve()),
        "cache_root": str(root / p["cache_dirname"]),
        "backbone_root": str(root / p["backbone_dirname"]),
        "backbone_exp_root": str(root / p["backbone_exp_dirname"]),
        "common_root": str(root / p["common_dirname"]),
        "supportbench_root": str(root / p["supportbench_dirname"]),
        "benchmark_root": str(root / p["benchmark_relpath"]),
        "exp_root": str(root / p["exp_dirname"]),
        "log_root": str(root / p["log_dirname"]),
        "contract": str(root / p["contract_relpath"]),
    }
    return cfg
