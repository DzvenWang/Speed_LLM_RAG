from __future__ import annotations

import torch
from torch import nn


def tod_dow_from_calendar_marks(x_mark: torch.Tensor, expected_history: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert canonical calendar marks to the TOD/DOW contract used by reference Backbones."""
    if x_mark.ndim != 3 or x_mark.shape[1] != expected_history or x_mark.shape[2] < 6:
        raise RuntimeError(
            f'Reference-backbone control time-mark contract failed: got {tuple(x_mark.shape)}, '
            f'expected [B,{expected_history},>=6]'
        )
    tod = torch.round((x_mark[..., 4] * 60.0 + x_mark[..., 5]) / 5.0).long().clamp_(0, 287)
    dow = torch.round(x_mark[..., 3]).long().clamp_(0, 6)
    return tod, dow


class ReferenceBackboneControlBase(nn.Module):
    """Common matched-backbone wrapper; ``frozen`` only needs predict_norm/adapter/checkpoint fields."""

    def __init__(self, native: nn.Module, frozen):
        super().__init__()
        self.native = native
        # Deliberately keep the frozen bridge as a plain helper, not a registered module.
        # This prevents reference-Backbone parameters from entering model.parameters()/optimizer.
        object.__setattr__(self, 'frozen', frozen)
        self.last_base_prediction: torch.Tensor | None = None
        direct = getattr(self.native, 'linear_x', None)
        if direct is not None:
            for p in direct.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen.adapter.model.eval()
        return self

    def patch_base(self, x: torch.Tensor, x_mark: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        H = int(x.shape[1])
        tod, dow = tod_dow_from_calendar_marks(x_mark, H)
        with torch.no_grad():
            base_abs = self.frozen.predict_norm(x, tod, dow)
        if base_abs.ndim != 3:
            raise RuntimeError(f'Reference Backbone output must be [B,P,N], got {tuple(base_abs.shape)}')
        x_offset = x[:, -1:, :].detach()
        if base_abs.shape[0] != x.shape[0] or base_abs.shape[2] != x.shape[2]:
            raise RuntimeError(
                f'Reference Backbone/base shape mismatch base={tuple(base_abs.shape)} x={tuple(x.shape)}'
            )
        self.last_base_prediction = base_abs.detach()
        return base_abs - x_offset, x_offset


class RAFTReferenceBackboneControl(ReferenceBackboneControlBase):
    """Official RAFT retrieval/fusion with a reference Backbone direct path.

    For strong pretrained Backbones such as STLAformer, the native RAFT fusion output
    can be badly mis-scaled relative to the reference forecast.  ``safe_residual``
    therefore turns the transplanted RAFT branch into a zero-initialized residual:

        final = base + alpha * (raft_fused - base)

    with learnable scalar ``alpha`` initialized to ``residual_scale_init`` (normally 0).
    This is a compatibility adapter only; the native RAFT retriever/fusion itself is
    otherwise unchanged.
    """

    def __init__(self, native: nn.Module, frozen, *, safe_residual: bool = False, residual_scale_init: float = 0.0):
        super().__init__(native, frozen)
        self.safe_residual = bool(safe_residual)
        if self.safe_residual:
            self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init), dtype=torch.float32))
        else:
            self.register_parameter('residual_scale', None)

    def residual_scale_value(self) -> float | None:
        if self.residual_scale is None:
            return None
        return float(self.residual_scale.detach().cpu().item())

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor, index: torch.Tensor, mode: str = 'train') -> torch.Tensor:
        n = self.native
        index = index.to(x.device)
        bsz, seq_len, channels = x.shape
        if seq_len != n.seq_len or channels != n.channels:
            raise RuntimeError(
                f'RAFT-reference input contract failed {(bsz,seq_len,channels)} '
                f'expected (*,{n.seq_len},{n.channels})'
            )
        base_rel, x_offset = self.patch_base(x, x_mark)

        pred_from_retrieval = n.retrieval_dict[mode][:, index].to(x.device)
        retrieval_pred_list = []
        for i, pr in enumerate(pred_from_retrieval):
            if tuple(pr.shape) != (bsz, n.pred_len, channels):
                raise RuntimeError(
                    f'RAFT-reference retrieval shape {tuple(pr.shape)} '
                    f'expected {(bsz,n.pred_len,channels)}'
                )
            g = n.period_num[i]
            pr = pr.reshape(bsz, n.pred_len // g, g, channels)[:, :, 0, :]
            pr = n.retrieval_pred[i](pr.permute(0, 2, 1)).permute(0, 2, 1)
            retrieval_pred_list.append(pr.reshape(bsz, n.pred_len, n.channels))
        retrieval_pred = torch.stack(retrieval_pred_list, dim=1).sum(dim=1)

        pred = torch.cat([base_rel, retrieval_pred], dim=1)
        pred = n.linear_pred(pred.permute(0, 2, 1)).permute(0, 2, 1)
        fused_abs = pred.reshape(bsz, n.pred_len, n.channels) + x_offset
        if not self.safe_residual:
            return fused_abs

        # Start exactly from the strong reference forecast.  The scalar can move away
        # from zero only when validation-supervised training finds useful RAFT signal.
        base_abs = base_rel + x_offset
        correction = fused_abs - base_abs
        return base_abs + self.residual_scale.to(dtype=correction.dtype) * correction


class SARAFReferenceBackboneControl(ReferenceBackboneControlBase):
    """Official SARAF retrieval/fusion with only linear_x replaced by the selected reference Backbone."""

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor, index: torch.Tensor, mode: str = 'train') -> torch.Tensor:
        n = self.native
        index = index.to(x.device)
        bsz, seq_len, channels = x.shape
        if seq_len != n.seq_len or channels != n.channels:
            raise RuntimeError(
                f'SARAF-reference input contract failed {(bsz,seq_len,channels)} '
                f'expected (*,{n.seq_len},{n.channels})'
            )
        base_rel, x_offset = self.patch_base(x, x_mark)

        preds_from_temporal_sims = n.retrieval_temporal_dict[mode][:, index].detach()
        if preds_from_temporal_sims.shape[0] < 1:
            raise RuntimeError('SARAF-reference retrieval dictionary is empty')
        pr = preds_from_temporal_sims[0]
        if tuple(pr.shape) != (bsz, n.pred_len, channels):
            raise RuntimeError(
                f'SARAF-reference retrieval shape {tuple(pr.shape)} '
                f'expected {(bsz,n.pred_len,channels)}'
            )
        retrieval_pred = n.retrieval_pred[0](pr.permute(0, 2, 1)).permute(0, 2, 1)
        pred = n.simple_fusion(base_rel, retrieval_pred)
        return pred.reshape(bsz, n.pred_len, n.channels) + x_offset


# Backward-compatible names retained for existing PatchSTG controlled experiments.
RAFTPatchSTGControl = RAFTReferenceBackboneControl
SARAFPatchSTGControl = SARAFReferenceBackboneControl
FrozenPatchSTGControlBase = ReferenceBackboneControlBase
