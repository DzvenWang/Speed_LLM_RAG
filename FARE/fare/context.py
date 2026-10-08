from __future__ import annotations

import os
from pathlib import Path
import torch

from .config import load_config
from .paths import install_repo_imports
from .utils import sha256_file


def discover_stla_checkpoint(root: Path, target: str, dataset: str, seed: int, explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists(): raise FileNotFoundError(p)
        return p
    base = root / "Backbone_exps" / target / "STLAformer" / dataset / "formal"
    if not base.exists(): raise FileNotFoundError(base)
    candidates = sorted(base.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        try: ck = torch.load(p, map_location="cpu", weights_only=False)
        except Exception: continue
        if str(ck.get("model", "STLAformer")) != "STLAformer": continue
        if str(ck.get("target", target)) != target: continue
        if int(ck.get("seed", seed)) != int(seed): continue
        return p
    raise RuntimeError(f"No compatible STLAformer checkpoint below {base}")


class FAREContext:
    def __init__(self, target: str, config_path=None, checkpoint: str | None = None, run_mode: str = "formal"):
        if target not in {"speed", "flow"}: raise ValueError(target)
        os.environ["TRAFFIDENT_TARGET"] = target
        self.root = install_repo_imports()
        self.cfg = load_config(config_path)
        self.target = target
        self.dataset = str(self.cfg["experiment"]["dataset"])
        self.seed = int(self.cfg["experiment"]["seed"])
        self.run_mode = str(run_mode)

        from backbone.utils import load_yaml as load_backbone_yaml
        from backbone.shared_cache import cache_path, validate_shared_cache
        from supportbench.dataset import SupportBench

        self.backbone_cfg = load_backbone_yaml(self.root / "Backbone" / "configs" / "backbone.yaml")
        if str(self.backbone_cfg["experiment"]["subgraph"]) != self.dataset:
            raise RuntimeError("FARE dataset != Backbone dataset")
        validate_shared_cache(self.backbone_cfg, "formal")
        self.formal_cache = cache_path(self.backbone_cfg, "formal")
        self.bench = SupportBench(self.dataset, root=self.backbone_cfg["project"]["benchmark_root"])
        self.checkpoint = discover_stla_checkpoint(self.root, target, self.dataset, self.seed, checkpoint)
        self.checkpoint_sha256 = sha256_file(self.checkpoint)
        self.ckpt = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if str(self.ckpt.get("target", target)) != target:
            raise RuntimeError("Backbone checkpoint target mismatch")
        self.mean = float(self.ckpt["mean"]); self.std = float(self.ckpt["std"])
        if self.std <= 0: raise RuntimeError("Invalid checkpoint std")

    @property
    def device(self):
        pref = str(self.cfg["runtime"].get("device", "auto"))
        if pref != "auto": return torch.device(pref)
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @property
    def artifact_root(self) -> Path:
        return Path(self.cfg["project"]["artifact_root"]) / self.target / self.dataset / "v0.2.4" / self.run_mode

    @property
    def exp_root(self) -> Path:
        return Path(self.cfg["project"]["exp_root"]) / self.target / "FARE" / self.dataset / "v0.2.4" / self.run_mode

    def evaluation_cache(self, severity: int) -> Path:
        from backbone.shared_cache import cache_path, validate_shared_cache
        validate_shared_cache(self.backbone_cfg, "evaluation", severity, allow_locked_evaluation=True)
        return cache_path(self.backbone_cfg, "evaluation", severity)

    def build_backbone(self):
        from backbone.adapters import build_adapter
        mc = dict(self.ckpt["model_config"])
        adapter = build_adapter(
            "STLAformer", self.root / "Backbone" / "vendor", int(self.bench.N), self.device,
            self.mean, self.std, mc, self.bench.adj, self.bench.nodes,
        )
        adapter.model.load_state_dict(self.ckpt["state_dict"], strict=True)
        adapter.model.eval()
        for p in adapter.model.parameters(): p.requires_grad_(False)
        return adapter
