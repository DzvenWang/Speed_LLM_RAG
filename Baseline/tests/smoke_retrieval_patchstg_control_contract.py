#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline.retrieval_patchstg_controls import (  # noqa: E402
    RAFTPatchSTGControl,
    SARAFPatchSTGControl,
    tod_dow_from_calendar_marks,
)


class FakeAdapter:
    def __init__(self):
        self.model = nn.Linear(1, 1)
        for p in self.model.parameters():
            p.requires_grad = False


class FakeFrozen:
    def __init__(self, pred_len: int):
        self.pred_len = pred_len
        self.adapter = FakeAdapter()
        self.checkpoint = Path('/tmp/frozen_patchstg.pt')
        self.reference_seed = 10

    @torch.no_grad()
    def predict_norm(self, x, tod, dow):
        # Deterministic nontrivial base forecast: repeat last observed value.
        assert tod.shape == x.shape[:2]
        assert dow.shape == x.shape[:2]
        return x[:, -1:, :].repeat(1, self.pred_len, 1)


class FakeRAFT(nn.Module):
    def __init__(self, H=4, P=4, N=3):
        super().__init__()
        self.seq_len = H
        self.pred_len = P
        self.channels = N
        self.period_num = [1]
        self.linear_x = nn.Linear(H, P)
        self.retrieval_pred = nn.ModuleList([nn.Linear(P, P, bias=False)])
        self.linear_pred = nn.Linear(2 * P, P, bias=False)
        with torch.no_grad():
            self.retrieval_pred[0].weight.copy_(torch.eye(P))
            self.linear_pred.weight.zero_()
            # Select PatchSTG relative path only: final should equal frozen PatchSTG.
            self.linear_pred.weight[:, :P].copy_(torch.eye(P))
        self.retrieval_dict = {
            'train': torch.randn(1, 8, P, N),
            'valid': torch.randn(1, 8, P, N),
        }


class FakeSARAF(nn.Module):
    def __init__(self, H=4, P=4, N=3):
        super().__init__()
        self.seq_len = H
        self.pred_len = P
        self.channels = N
        self.linear_x = nn.Linear(H, P)
        self.retrieval_pred = nn.ModuleList([nn.Linear(P, P, bias=False)])
        self.linear_pred_2 = nn.Linear(P, P, bias=False)
        with torch.no_grad():
            self.retrieval_pred[0].weight.copy_(torch.eye(P))
            self.linear_pred_2.weight.copy_(torch.eye(P))
        self.retrieval_temporal_dict = {
            'train': torch.zeros(1, 8, P, N),
            'valid': torch.zeros(1, 8, P, N),
        }

    def simple_fusion(self, x_pred_from_x, preds_from_temporal_sims):
        pred = 0.5 * x_pred_from_x + 0.5 * preds_from_temporal_sims
        return self.linear_pred_2(pred.permute(0, 2, 1)).permute(0, 2, 1)


def marks(B, H):
    out = torch.zeros(B, H, 6)
    out[..., 3] = 2
    for t in range(H):
        out[:, t, 4] = 8
        out[:, t, 5] = t * 5
    return out


def main():
    B, H, P, N = 2, 4, 4, 3
    x = torch.randn(B, H, N)
    xm = marks(B, H)
    idx = torch.tensor([0, 1], dtype=torch.long)

    tod, dow = tod_dow_from_calendar_marks(xm, H)
    assert torch.equal(tod[0], torch.tensor([96, 97, 98, 99]))
    assert torch.equal(dow[0], torch.tensor([2, 2, 2, 2]))

    frozen = FakeFrozen(P)
    raft_native = FakeRAFT(H, P, N)
    raft = RAFTPatchSTGControl(raft_native, frozen)
    raft_out = raft(x, xm, idx, mode='valid')
    base = frozen.predict_norm(x, tod, dow)
    torch.testing.assert_close(raft_out, base, atol=1e-6, rtol=0)
    assert not any(p.requires_grad for p in raft_native.linear_x.parameters())
    assert all(p.requires_grad for p in raft_native.linear_pred.parameters())

    saraf_native = FakeSARAF(H, P, N)
    saraf = SARAFPatchSTGControl(saraf_native, frozen)
    saraf_out = saraf(x, xm, idx, mode='valid')
    # Retrieval is zero in offset-relative space, so official 0.5 fusion should
    # halve the PatchSTG relative component before adding x_offset back.
    x_offset = x[:, -1:, :]
    expected = 0.5 * (base - x_offset) + x_offset
    torch.testing.assert_close(saraf_out, expected, atol=1e-6, rtol=0)
    assert not any(p.requires_grad for p in saraf_native.linear_x.parameters())
    assert all(p.requires_grad for p in saraf_native.linear_pred_2.parameters())

    print('RETRIEVAL-PATCHSTG CONTROL CONTRACT PASS')
    print({
        'raft_shape': tuple(raft_out.shape),
        'saraf_shape': tuple(saraf_out.shape),
        'tod': tod[0].tolist(),
        'dow': dow[0].tolist(),
        'direct_paths_frozen': True,
    })


if __name__ == '__main__':
    main()
