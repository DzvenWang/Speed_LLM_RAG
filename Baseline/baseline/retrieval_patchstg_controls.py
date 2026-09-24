from __future__ import annotations

import torch
from torch import nn


def tod_dow_from_calendar_marks(x_mark: torch.Tensor, expected_history: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert canonical [year,month,day,dow,hour,minute] marks to PatchSTG TOD/DOW."""
    if x_mark.ndim != 3 or x_mark.shape[1] != expected_history or x_mark.shape[2] < 6:
        raise RuntimeError(
            f'PatchSTG control time-mark contract failed: got {tuple(x_mark.shape)}, '
            f'expected [B,{expected_history},>=6]'
        )
    tod = torch.round((x_mark[..., 4] * 60.0 + x_mark[..., 5]) / 5.0).long().clamp_(0, 287)
    dow = torch.round(x_mark[..., 3]).long().clamp_(0, 6)
    return tod, dow


class FrozenPatchSTGControlBase(nn.Module):
    """Common matched-backbone wrapper; ``frozen`` only needs predict_norm/adapter/checkpoint fields."""

    def __init__(self, native: nn.Module, frozen):
        super().__init__()
        self.native = native
        # Deliberately keep the frozen bridge as a plain helper, not a registered module.
        # This prevents PatchSTG parameters from entering model.parameters()/optimizer.
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
            raise RuntimeError(f'Frozen PatchSTG output must be [B,P,N], got {tuple(base_abs.shape)}')
        x_offset = x[:, -1:, :].detach()
        if base_abs.shape[0] != x.shape[0] or base_abs.shape[2] != x.shape[2]:
            raise RuntimeError(
                f'Frozen PatchSTG/base shape mismatch base={tuple(base_abs.shape)} x={tuple(x.shape)}'
            )
        self.last_base_prediction = base_abs.detach()
        return base_abs - x_offset, x_offset


class RAFTPatchSTGControl(FrozenPatchSTGControlBase):
    """Official RAFT retrieval/fusion with only linear_x replaced by Frozen PatchSTG."""

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor, index: torch.Tensor, mode: str = 'train') -> torch.Tensor:
        n = self.native
        index = index.to(x.device)
        bsz, seq_len, channels = x.shape
        if seq_len != n.seq_len or channels != n.channels:
            raise RuntimeError(
                f'RAFT-PatchSTG input contract failed {(bsz,seq_len,channels)} '
                f'expected (*,{n.seq_len},{n.channels})'
            )
        base_rel, x_offset = self.patch_base(x, x_mark)

        pred_from_retrieval = n.retrieval_dict[mode][:, index].to(x.device)
        retrieval_pred_list = []
        for i, pr in enumerate(pred_from_retrieval):
            if tuple(pr.shape) != (bsz, n.pred_len, channels):
                raise RuntimeError(
                    f'RAFT-PatchSTG retrieval shape {tuple(pr.shape)} '
                    f'expected {(bsz,n.pred_len,channels)}'
                )
            g = n.period_num[i]
            pr = pr.reshape(bsz, n.pred_len // g, g, channels)[:, :, 0, :]
            pr = n.retrieval_pred[i](pr.permute(0, 2, 1)).permute(0, 2, 1)
            retrieval_pred_list.append(pr.reshape(bsz, n.pred_len, n.channels))
        retrieval_pred = torch.stack(retrieval_pred_list, dim=1).sum(dim=1)

        pred = torch.cat([base_rel, retrieval_pred], dim=1)
        pred = n.linear_pred(pred.permute(0, 2, 1)).permute(0, 2, 1)
        return pred.reshape(bsz, n.pred_len, n.channels) + x_offset


class SARAFPatchSTGControl(FrozenPatchSTGControlBase):
    """Official SARAF retrieval/fusion with only linear_x replaced by Frozen PatchSTG."""

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor, index: torch.Tensor, mode: str = 'train') -> torch.Tensor:
        n = self.native
        index = index.to(x.device)
        bsz, seq_len, channels = x.shape
        if seq_len != n.seq_len or channels != n.channels:
            raise RuntimeError(
                f'SARAF-PatchSTG input contract failed {(bsz,seq_len,channels)} '
                f'expected (*,{n.seq_len},{n.channels})'
            )
        base_rel, x_offset = self.patch_base(x, x_mark)

        preds_from_temporal_sims = n.retrieval_temporal_dict[mode][:, index].detach()
        if preds_from_temporal_sims.shape[0] < 1:
            raise RuntimeError('SARAF-PatchSTG retrieval dictionary is empty')
        pr = preds_from_temporal_sims[0]
        if tuple(pr.shape) != (bsz, n.pred_len, channels):
            raise RuntimeError(
                f'SARAF-PatchSTG retrieval shape {tuple(pr.shape)} '
                f'expected {(bsz,n.pred_len,channels)}'
            )
        retrieval_pred = n.retrieval_pred[0](pr.permute(0, 2, 1)).permute(0, 2, 1)
        pred = n.simple_fusion(base_rel, retrieval_pred)
        return pred.reshape(bsz, n.pred_len, n.channels) + x_offset
