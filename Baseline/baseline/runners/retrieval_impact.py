from __future__ import annotations

import numpy as np
import torch


IMPACT_KEYS = (
    'count',
    'correction_abs_mean_norm',
    'correction_rms_norm',
    'correction_max_abs_norm',
    'helpful_fraction',
    'harmful_fraction',
    'neutral_fraction',
    'mean_abs_error_change_norm',
    'base_MAE_norm',
    'final_MAE_norm',
)


def empty_impact_stats() -> dict:
    return {
        'count': 0,
        'corr_abs_sum': 0.0,
        'corr_sq_sum': 0.0,
        'corr_max_abs': 0.0,
        'helpful': 0,
        'harmful': 0,
        'neutral': 0,
        'delta_error_sum': 0.0,
        'base_abs_error_sum': 0.0,
        'final_abs_error_sum': 0.0,
    }


def accumulate_impact(
    stats: dict,
    final: torch.Tensor,
    base: torch.Tensor,
    y: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    """Accumulate matched-backbone retrieval impact on valid forecast targets.

    This intentionally reproduces the RAFT-/SARAF-PatchSTG definition:
    helpful iff |final-y| < |base-y| by >1e-7, harmful iff larger by >1e-7,
    otherwise neutral. All quantities remain in normalized space.
    """
    if tuple(final.shape) != tuple(base.shape) or tuple(final.shape) != tuple(y.shape):
        raise RuntimeError(
            'Matched PatchSTG impact contract failed: '
            f'final={tuple(final.shape)} base={tuple(base.shape)} y={tuple(y.shape)}'
        )
    mask = valid.bool()
    if not torch.any(mask):
        return
    corr = (final.detach() - base.detach())[mask]
    base_err = (base.detach() - y.detach()).abs()[mask]
    final_err = (final.detach() - y.detach()).abs()[mask]
    delta_err = final_err - base_err
    eps = 1e-7
    stats['count'] += int(corr.numel())
    stats['corr_abs_sum'] += float(corr.abs().sum().item())
    stats['corr_sq_sum'] += float((corr.double() * corr.double()).sum().item())
    stats['corr_max_abs'] = max(stats['corr_max_abs'], float(corr.abs().max().item()))
    stats['helpful'] += int((delta_err < -eps).sum().item())
    stats['harmful'] += int((delta_err > eps).sum().item())
    stats['neutral'] += int((delta_err.abs() <= eps).sum().item())
    stats['delta_error_sum'] += float(delta_err.sum().item())
    stats['base_abs_error_sum'] += float(base_err.sum().item())
    stats['final_abs_error_sum'] += float(final_err.sum().item())


def finalize_impact(stats: dict) -> dict | None:
    if int(stats['count']) <= 0:
        return None
    n = int(stats['count'])
    return {
        'count': n,
        'correction_abs_mean_norm': float(stats['corr_abs_sum'] / n),
        'correction_rms_norm': float(np.sqrt(stats['corr_sq_sum'] / n)),
        'correction_max_abs_norm': float(stats['corr_max_abs']),
        'helpful_fraction': float(stats['helpful'] / n),
        'harmful_fraction': float(stats['harmful'] / n),
        'neutral_fraction': float(stats['neutral'] / n),
        # Negative means the retrieval control improved on frozen PatchSTG.
        'mean_abs_error_change_norm': float(stats['delta_error_sum'] / n),
        'base_MAE_norm': float(stats['base_abs_error_sum'] / n),
        'final_MAE_norm': float(stats['final_abs_error_sum'] / n),
    }
