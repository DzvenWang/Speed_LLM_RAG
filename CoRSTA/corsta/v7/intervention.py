"""Only delete available history values; partition synthetic faults deterministically.

No access to future labels. Each mask shape is [B,H,N].
"""
from __future__ import annotations
import torch


def fault_partition(observed: torch.Tensor, reference: torch.Tensor | None = None, groups: int = 2):
    if observed.ndim != 3 or not 2 <= groups <= 3:
        raise ValueError('Expected [B,H,N] mask and 2..3 groups')
    source = torch.ones_like(observed) if reference is None else reference
    if source.shape != observed.shape:
        raise ValueError('Source/observed mask shapes disagree')
    if ((observed > .5) & (source < .5)).any():
        raise ValueError('Cannot reveal a source-unavailable measurement')
    missing = (source > .5) & (observed < .5)
    b, h, n = missing.shape
    # Assign whole sensor histories, not arbitrary future-dependent positions.
    ids = torch.arange(n,device=missing.device).remainder(groups)
    group_masks = torch.stack([missing & (ids == j).view(1,1,n) for j in range(groups)],dim=1)
    node_masks = group_masks.any(dim=2)  # [B,J,N]
    return group_masks, node_masks


def teacher_view(x_reference: torch.Tensor, reference_mask: torch.Tensor, group_fault: torch.Tensor):
    m = reference_mask * (~group_fault).to(reference_mask.dtype)
    return x_reference * m, m


def decompose_response(y_reference, y_joint, y_single, node_masks):
    """y_single [B,J,P,N], node_masks [B,J,N]. Exact additive partition."""
    if y_single.ndim != 4 or y_single.shape[0] != y_reference.shape[0] or y_single.shape[2:] != y_reference.shape[1:]:
        raise ValueError('Bad singleton predictions')
    if node_masks.shape != (y_single.shape[0],y_single.shape[1],y_single.shape[-1]):
        raise ValueError('Bad source-node masks')
    d = y_reference.unsqueeze(1) - y_single
    local = (d * node_masks[:, :, None, :]).sum(dim=1)
    nonlocal_response = (d * (~node_masks)[:, :, None, :]).sum(dim=1)
    interaction = (y_reference-y_joint) - d.sum(dim=1)
    return {'c1':local,'c2':nonlocal_response,'interaction':interaction,
            'total':y_reference-y_joint}
