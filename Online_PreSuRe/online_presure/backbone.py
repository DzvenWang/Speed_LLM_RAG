from __future__ import annotations
from pathlib import Path
import json
import torch
from .contract import load_cache_identity, compare_checkpoint_identity
from .utils import add_sys_path, sha256_file, log


def _resolve_dev_checkpoint(cfg):
    root = Path(cfg["project"]["backbone_exp_root"]) / "PatchSTG" / cfg["experiment"]["subgraph"] / "screening"
    target = load_cache_identity(cfg, "development")
    cand, rejected = [], []
    for s in root.glob("*/summary.json") if root.exists() else []:
        try:
            j = json.loads(s.read_text(encoding="utf-8")); ck = s.parent / "best.pt"
            if int(j.get("seed", -1)) != int(cfg["experiment"]["reference_seed"]) or not ck.exists(): continue
            if j.get("model") != "PatchSTG" or j.get("stage") != "screening": continue
            obj = torch.load(ck, map_location="cpu", weights_only=False)
            compare_checkpoint_identity(obj, target, "development")
            cand.append((s.stat().st_mtime, ck))
        except Exception as e:
            rejected.append((str(s), str(e)))
    if not cand:
        msg = "No completed current-contract PatchSTG screening seed10 checkpoint found."
        if rejected: msg += f" Rejected {len(rejected)} stale/mismatched run(s)."
        raise RuntimeError(msg)
    return max(cand, key=lambda x: x[0])[1], "screening_seed10_current_contract"


def resolve_checkpoint(cfg, stage, override=None):
    if override:
        p, source = Path(override), "explicit_override"
    elif stage == "development":
        p, source = _resolve_dev_checkpoint(cfg)
    elif stage == "formal":
        f = Path(cfg["project"]["backbone_exp_root"]) / "FROZEN_BACKBONE.json"
        if not f.exists(): raise RuntimeError("Formal online run requires Backbone_exps/FROZEN_BACKBONE.json")
        j = json.loads(f.read_text(encoding="utf-8"))
        if j.get("selected_model") != "PatchSTG" or int(j.get("reference_seed", -1)) != int(cfg["experiment"]["reference_seed"]):
            raise RuntimeError("Frozen backbone is not configured PatchSTG reference seed10")
        p, source = Path(j["reference_checkpoint"]), "FROZEN_BACKBONE.json"
        if sha256_file(p) != str(j.get("reference_checkpoint_sha256", "")):
            raise RuntimeError("Frozen checkpoint SHA256 mismatch")
    else:
        raise ValueError(stage)
    if not p.exists(): raise FileNotFoundError(p)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if ck.get("model") != "PatchSTG": raise RuntimeError(f"Expected PatchSTG checkpoint, got {ck.get('model')}")
    if int(ck.get("seed", -1)) != int(cfg["experiment"]["reference_seed"]):
        raise RuntimeError("Online_PSTG requires fixed PatchSTG reference seed10")
    compare_checkpoint_identity(ck, load_cache_identity(cfg, stage), stage)
    return p, ck, source


def load_patchstg(cfg, stage, device, override=None):
    add_sys_path(cfg["project"]["backbone_root"])
    add_sys_path(cfg["project"]["supportbench_root"])
    from backbone.adapters import build_adapter
    from supportbench.dataset import SupportBench
    path, ck, source = resolve_checkpoint(cfg, stage, override)
    b = SupportBench(cfg["experiment"]["subgraph"], root=cfg["project"]["benchmark_root"])
    adapter = build_adapter(
        "PatchSTG", Path(cfg["project"]["backbone_root"]) / "vendor", b.N, device,
        float(ck["mean"]), float(ck["std"]), ck["model_config"], b.adj, b.nodes,
    )
    adapter.model.load_state_dict(ck["state_dict"]); adapter.model.eval()
    log(f"PATCHSTG checkpoint={path} source={source} seed={ck['seed']} best_epoch={ck.get('best_epoch')}")
    return adapter.model, path, ck, source
