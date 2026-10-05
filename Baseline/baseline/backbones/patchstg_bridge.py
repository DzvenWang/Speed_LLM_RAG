from __future__ import annotations
import torch
import torch.nn as nn
from .reference_backbone import ReferenceBackbone, ReferenceBackboneRASTBridge

class FrozenPatchSTG(ReferenceBackbone):
    """Backward-compatible PatchSTG-specific alias.

    No FROZEN_BACKBONE manifest is used. The current compatible PatchSTG Formal
    checkpoint is resolved directly unless an exact checkpoint is supplied.
    """
    def __init__(self,cfg:dict,device:torch.device,checkpoint=None,seed:int=10):
        super().__init__(cfg,device,model='PatchSTG',seed=seed,checkpoint=checkpoint)

class PatchSTGRASTBridge(ReferenceBackboneRASTBridge):
    """Backward-compatible alias for the generic RAST reference-backbone bridge."""
    def __init__(self, frozen: FrozenPatchSTG):
        super().__init__(frozen)
        self.frozen = frozen
        self.patchstg = frozen.adapter.model
