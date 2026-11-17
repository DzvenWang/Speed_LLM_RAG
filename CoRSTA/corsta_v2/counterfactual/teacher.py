"""Paired full/observed teacher targets from identical frozen STLA weights.

All counterfactual/residual target arithmetic is intentionally FP32, even when
this module is called from a mixed-precision (AMP) training forward. The STLA
outputs may be FP16; subtracting them before the cast loses cancellation
precision and can spuriously fail the exact decomposition check.
"""
from __future__ import annotations
import torch
from torch import nn
from corsta_v2.stla.backbone import base_features


class PairedTeacher(nn.Module):
    def __init__(self, frozen_stla):
        super().__init__()
        self.stla = frozen_stla
        self.stla.requires_grad_(False)
        self.stla.eval()

    def train(self, mode=True):
        super().train(False)
        self.stla.eval()
        return self

    @torch.no_grad()
    def forward(self, batch, observed_h0=None, observed_p0=None):
        if 'x_full' not in batch or 'original_valid' not in batch:
            raise ValueError('Paired Teacher forbidden outside TRAIN')

        hf, _, pf = base_features(
            self.stla, batch['x_full'], batch['original_valid'],
            batch['tod'], batch['dow'],
        )
        if observed_h0 is None or observed_p0 is None:
            hm, _, pm = base_features(
                self.stla, batch['x'], batch['mask'],
                batch['tod'], batch['dow'],
            )
        else:
            hm, pm = observed_h0.detach(), observed_p0.detach()

        # AMP may produce FP16 STLA predictions and hidden features. Converting
        # *after* subtraction would preserve a lossy FP16 CF target and fail
        # the identity even though both forecasts and Y are valid.
        pf = pf.float()
        pm = pm.float()
        hf = hf.float()
        hm = hm.float()
        y_src = batch['y_norm'].float()
        valid = batch['future_valid'].bool()
        y = torch.where(valid, y_src, torch.zeros_like(y_src))

        # Do not conceal a real data/model error by loosening allclose or
        # declaring NaN == NaN. Invalid future labels were masked above;
        # predictions must still be finite to be used as supervision.
        for name, tensor in (('p_full', pf), ('p_masked', pm), ('valid_y', y)):
            if not torch.isfinite(tensor).all():
                n_bad = int((~torch.isfinite(tensor)).sum().item())
                raise RuntimeError(
                    f'PairedTeacher non-finite {name}: count={n_bad} '
                    f'dtype={tensor.dtype} shape={tuple(tensor.shape)}'
                )

        cf = pf - pm
        residual = y - pf
        total = y - pm
        reconstructed = cf + residual

        # For values represented in FP32, FP32 add/subtract can incur a few
        # ULPs of cancellation. This scale-aware bound accepts such roundoff
        # but detects inconsistencies larger than FP32's numerical precision.
        eps = torch.finfo(torch.float32).eps
        tolerance = 2e-6 + 8.0 * eps * (pf.abs() + pm.abs() + y.abs())
        delta = (reconstructed - total).abs()
        if not torch.all(delta <= tolerance):
            bad = delta > tolerance
            raise RuntimeError(
                'CF/Residual target decomposition broken after FP32 casting: '
                f'bad={int(bad.sum().item())} '
                f'max_abs_diff={float(delta.max().item()):.8g} '
                f'max_tolerance={float(tolerance.max().item()):.8g} '
                f'pf_dtype={pf.dtype} pm_dtype={pm.dtype}'
            )

        return {
            'p_full': pf.detach(),
            'p_masked': pm.detach(),
            'cf_target': cf.detach(),
            'residual_target': residual.detach(),
            'latent_cf_target': (hf - hm).detach(),
            'cf_utility': ((pm - y).abs() - (pf - y).abs()).detach(),
        }
