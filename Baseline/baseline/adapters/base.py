from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class AdapterContext:
    project_root: Path
    cache_root: Path
    exp_root: Path
    log_root: Path
    backbone_model: str | None = None
    backbone_checkpoint: Path | None = None

class OfficialAdapter:
    method = None
    def __init__(self, ctx: AdapterContext): self.ctx=ctx
    def compatibility_audit(self): raise NotImplementedError
    def prepare_bridge(self): raise NotImplementedError
    def run_command(self, phase: str): raise NotImplementedError
