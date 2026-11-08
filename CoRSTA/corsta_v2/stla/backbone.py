"""Official STLA + mask projection with validated v0.1 S0 initialization.
Only STLA source, loader and split wrapper are reused; no retrieval runtime.
"""
from __future__ import annotations
from pathlib import Path
import torch
from corsta.stla import load_pretrained_stla
from corsta.student import SplitSTLA
from corsta_v2.util import sha256,checkpoint_load


def make_stla(root,backbone_checkpoint,mask_checkpoint,device,tail_layers=2):
    base,bc=load_pretrained_stla(Path(root),Path(backbone_checkpoint),device)
    stla=SplitSTLA(base,tail_layers=tail_layers).to(device)
    if mask_checkpoint is None:
        raise RuntimeError('CoRSTA v0.2 requires a verified S0 mask_proj checkpoint; no implicit zero-mask projector')
    s0=checkpoint_load(mask_checkpoint)
    expected=sha256(backbone_checkpoint)
    if s0.get('backbone_sha256')!=expected:
        raise RuntimeError('S0 mask projector belongs to a different STLA checkpoint')
    stla.mask_proj.load_state_dict(s0['mask_proj'],strict=True)
    for p in stla.parameters():p.requires_grad_(False)
    stla.eval()
    return stla,bc


def base_features(stla,x,mask,tod,dow):
    with torch.no_grad():
        h0,context=stla.encode_prefix(x,mask,tod,dow)
        p0=stla.forecast_head(stla.encode_tail(h0,context))
    return h0,context,p0


def decode(stla,hidden,context):
    """Fixed STLA params, but differentiable wrt input hidden state."""
    return stla.forecast_head(stla.encode_tail(hidden,context))
