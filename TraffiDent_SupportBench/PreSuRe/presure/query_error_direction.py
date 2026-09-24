from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .models import NodeSensingEncoder
from .need_state import binary_metrics
from .probes import exact_spearman_sample, pearson


class QueryFutureErrorDirectionNet(nn.Module):
    """Query-only probe for the future PatchSTG correction demand.

    Inputs are the deployment-safe node feature tensor produced by PreSuRe:
    incomplete history, history observation masks, graph-neighbour context,
    time covariates, sensing-state descriptors and the frozen PatchSTG forecast.
    No retrieval candidate, historical support residual, Oracle choice or current
    query future target is an input.

    Outputs:
      error_mu  [B,P,N] : signed future PatchSTG error in normalized-speed units.
      dir_logit [B,P,N] : P(Y_future - Y_base > 0), i.e. PatchSTG underprediction.
    """

    def __init__(
        self,
        din: int,
        num_nodes: int,
        P: int,
        base_start: int,
        sense_dim: int = 6,
        encoder_hidden: int = 192,
        state_dim: int = 96,
        node_dim: int = 24,
        horizon_dim: int = 24,
        hidden: int = 192,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.P = int(P)
        self.N = int(num_nodes)
        self.F = int(din)
        self.base_start = int(base_start)
        self.sense_dim = int(sense_dim)
        self.encoder = NodeSensingEncoder(
            self.F, self.N, hidden=int(encoder_hidden), demb=int(state_dim),
            node_dim=int(node_dim), dropout=float(dropout),
        )
        self.h_emb = nn.Embedding(self.P, int(horizon_dim))
        # local forecast descriptors: base_h, backward/forward deltas, curvature.
        local_dim = 4 + self.sense_dim
        self.trunk = nn.Sequential(
            nn.Linear(int(state_dim) + int(horizon_dim) + local_dim, int(hidden)),
            nn.LayerNorm(int(hidden)), nn.GELU(), nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), 96), nn.GELU(),
        )
        self.error_head = nn.Linear(96, 1)
        self.direction_head = nn.Linear(96, 1)

    def _local_forecast_features(self, x: torch.Tensor) -> torch.Tensor:
        # x [B,N,F] -> [B,N,P,4+sense_dim]
        base = x[:, :, self.base_start:self.base_start + self.P]
        if base.shape[-1] != self.P:
            raise RuntimeError(
                f'Cannot recover PatchSTG trajectory from feature shape {tuple(x.shape)}; '
                f'base_start={self.base_start}, P={self.P}'
            )
        prev = torch.cat([base[:, :, :1], base[:, :, :-1]], dim=-1)
        nxt = torch.cat([base[:, :, 1:], base[:, :, -1:]], dim=-1)
        dprev = base - prev
        dnext = nxt - base
        curve = dnext - dprev
        sense = x[:, :, -self.sense_dim:]
        sense = sense[:, :, None, :].expand(-1, -1, self.P, -1)
        local = torch.stack([base, dprev, dnext, curve], dim=-1)
        return torch.cat([local, sense], dim=-1)

    def forward_features(self, x: torch.Tensor):
        B, N, _ = x.shape
        if N != self.N:
            raise ValueError(f'Node count mismatch {N} != {self.N}')
        z = self.encoder(x)
        ze = z[:, :, None, :].expand(-1, -1, self.P, -1)
        he = self.h_emb.weight[None, None, :, :].expand(B, N, -1, -1)
        local = self._local_forecast_features(x)
        h = self.trunk(torch.cat([ze, he, local], dim=-1))
        return h

    def forward(self, x: torch.Tensor):
        h = self.forward_features(x)
        err = self.error_head(h).squeeze(-1).permute(0, 2, 1).contiguous()
        logit = self.direction_head(h).squeeze(-1).permute(0, 2, 1).contiguous()
        return err, logit


def derive_abs_error_thresholds(error_raw, valid, quantiles=(0.5, 0.75, 0.9)):
    """Horizon-specific thresholds from Train only."""
    e = np.asarray(error_raw, np.float32)
    v = np.asarray(valid, bool)
    if e.shape != v.shape or e.ndim != 3:
        raise ValueError(f'Expected matching [B,P,N] arrays, got {e.shape} and {v.shape}')
    out = {}
    for q in quantiles:
        arr = []
        for h in range(e.shape[1]):
            z = np.abs(e[:, h, :][v[:, h, :]])
            if len(z) == 0:
                raise RuntimeError(f'No valid Train cells at horizon {h}')
            arr.append(float(np.quantile(z, float(q))))
        out[f'q{int(round(100*float(q))):02d}'] = arr
    return out


def threshold_array(thresholds, name):
    return np.asarray(thresholds[name], np.float32)[None, :, None]


def query_direction_loss(
    error_mu_norm: torch.Tensor,
    direction_logit: torch.Tensor,
    error_raw: torch.Tensor,
    valid: torch.Tensor,
    speed_std: float,
    q50_raw: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    huber_beta: float = 0.25,
    regression_weight: float = 0.6,
    direction_weight: float = 1.0,
    magnitude_weight: float = 0.5,
    target_weight_clip: float = 3.0,
):
    target_norm = error_raw / max(float(speed_std), 1e-6)
    mag = torch.abs(target_norm)
    w = 1.0 + float(magnitude_weight) * torch.clamp(mag, max=float(target_weight_clip))
    reg = torch.nn.functional.smooth_l1_loss(
        error_mu_norm, target_norm, beta=float(huber_beta), reduction='none'
    )
    reg = (reg * w * valid.float()).sum() / valid.float().sum().clamp_min(1.0)

    eligible = valid & (torch.abs(error_raw) >= q50_raw)
    target_dir = (error_raw > 0).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        direction_logit, target_dir, reduction='none', pos_weight=pos_weight
    )
    dw = w * eligible.float()
    direction = (bce * dw).sum() / dw.sum().clamp_min(1.0)
    total = float(regression_weight) * reg + float(direction_weight) * direction
    return total, {
        'regression': float(reg.detach().cpu()),
        'direction': float(direction.detach().cpu()),
        'eligible_fraction': float(eligible.float().mean().detach().cpu()),
    }


def _view_mask(ds, ids, view, P):
    if view == 'overall':
        return np.ones((len(ids), P, np.asarray(ds.valid[ids]).shape[-1]), bool)
    if view == 'masked':
        return np.asarray(ds.direct[ids], bool)[:, None, :].repeat(P, axis=1)
    if view == 'support':
        return np.asarray(ds.support[ids], bool)[:, None, :].repeat(P, axis=1)
    raise ValueError(view)


def evaluate_direction_probe(
    pred_error_norm,
    direction_prob,
    error_raw,
    ds,
    ids,
    thresholds,
    speed_std,
    metric_sample_points=500000,
    seed=10,
    direction_threshold=0.5,
):
    pred_error_raw = np.asarray(pred_error_norm, np.float32) * float(speed_std)
    prob = np.asarray(direction_prob, np.float32)
    err = np.asarray(error_raw, np.float32)
    valid = np.asarray(ds.valid[ids], bool)
    P = err.shape[1]
    out = {'views': {}, 'masked_by_horizon': {}}
    for vi, view in enumerate(['overall', 'masked', 'support']):
        vm = valid & _view_mask(ds, ids, view, P)
        z = vm
        reg = {
            'count': int(z.sum()),
            'mae_raw': float(np.mean(np.abs(pred_error_raw[z] - err[z]))) if np.any(z) else None,
            'pearson': pearson(pred_error_raw[z], err[z]) if np.any(z) else None,
            'spearman': exact_spearman_sample(
                pred_error_raw[z], err[z], max_points=metric_sample_points, seed=seed + 31*vi
            ) if np.any(z) else None,
        }
        view_out = {'regression': reg}
        for qi, name in enumerate(['q50', 'q75', 'q90']):
            use = vm & (np.abs(err) >= threshold_array(thresholds, name))
            met = binary_metrics(
                prob[use], err[use] > 0, threshold=float(direction_threshold),
                max_points=metric_sample_points, seed=seed + 100*vi + qi,
            ) if np.any(use) else {}
            met['count'] = int(use.sum())
            # direct sign from regression head is a useful independent diagnostic.
            if np.any(use):
                met['regression_sign_accuracy'] = float(np.mean((pred_error_raw[use] > 0) == (err[use] > 0)))
            else:
                met['regression_sign_accuracy'] = None
            view_out[name] = met
        out['views'][view] = view_out

    masked = valid & _view_mask(ds, ids, 'masked', P)
    for h in range(P):
        row = {}
        for qi, name in enumerate(['q75', 'q90']):
            th = float(thresholds[name][h])
            use = masked[:, h, :] & (np.abs(err[:, h, :]) >= th)
            if np.any(use):
                met = binary_metrics(
                    prob[:, h, :][use], err[:, h, :][use] > 0, threshold=float(direction_threshold),
                    max_points=metric_sample_points, seed=seed + 500 + 17*h + qi,
                )
                row[name] = {
                    'count': int(use.sum()), 'auroc': met.get('auroc'),
                    'balanced_accuracy': met.get('balanced_accuracy'),
                    'accuracy': met.get('accuracy'), 'prevalence': met.get('prevalence'),
                }
            else:
                row[name] = {'count': 0, 'auroc': None, 'balanced_accuracy': None, 'accuracy': None, 'prevalence': None}
        out['masked_by_horizon'][f'h{h+1:02d}'] = row
    return out
