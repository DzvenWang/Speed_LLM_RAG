"""Lightweight, label-safe FARE training diagnostics.

Computes diagnostic *aggregates* only on sampled train/validation batches.
No extra model forward, candidate retrieval, or CPU synchronisation per batch.
All loss/correction labels below are used for monitoring, not inference.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch


def _safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


class OnlineDiagnostics:
    """On-device (numerator, denominator) aggregates, synced once per epoch."""

    def __init__(self, every_batches: int = 50, max_batches: int = 160):
        self.every_batches = int(every_batches)
        self.max_batches = int(max_batches)
        if self.every_batches < 1 or self.max_batches < 0:
            raise ValueError('every_batches must be >= 1; max_batches must be >= 0')
        self.values: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.sampled_batches = 0
        self.sampled_examples = 0

    def should_sample(self, batch_index: int) -> bool:
        return (batch_index % self.every_batches == 0 and
                (self.max_batches == 0 or self.sampled_batches < self.max_batches))

    def _add(self, name: str, values: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        x = values.detach().float()
        if mask is None:
            numerator, denominator = x.sum(), x.new_tensor(float(x.numel()))
        else:
            weight = mask.detach().to(device=x.device, dtype=x.dtype)
            # Explicit broadcast allows singleton batch/node/retrieval dimensions.
            numerator = (x * weight).sum()
            denominator = weight.expand_as(x).sum()
        if name in self.values:
            num, den = self.values[name]
            self.values[name] = (num + numerator, den + denominator)
        else:
            self.values[name] = (numerator, denominator)

    @torch.no_grad()
    def update(self, batch: dict[str, Any], output: dict[str, Any], mean: float, std: float,
               grad_norm: torch.Tensor | None = None) -> None:
        self.sampled_batches += 1
        self.sampled_examples += len(batch['x_obs'])
        self._add('historical_observed_fraction', batch['obs_mask'].float())
        if grad_norm is not None:
            self._add('preclip_gradient_norm', grad_norm.reshape(1))
        if 'retrieval_valid' in output:
            self._add('retrieval_valid_group_fraction', output['retrieval_valid'].float())
        if 'retrieved_valid_candidates' in output:
            self._add('retrieved_candidate_valid_fraction', output['retrieved_valid_candidates'].float())
        if 'retrieved_weights' in output:
            weights = output['retrieved_weights'].detach().float().clamp_min(0)
            valid = output.get('retrieval_valid',weights.sum(-1)>0).bool()
            sums = weights.sum(-1).clamp_min(1e-9)
            normalized = weights / sums.unsqueeze(-1)
            entropy = -(normalized * normalized.clamp_min(1e-9).log()).sum(-1)
            effective_k = 1.0 / normalized.square().sum(-1).clamp_min(1e-9)
            self._add('retrieval_weight_entropy', entropy, valid)
            self._add('retrieval_effective_topk', effective_k, valid)
            self._add('retrieval_top1_weight', normalized.max(-1).values, valid)
        for key, field in (
            ('state_uncertainty_mean','state_uncertainty'),
            ('innovation_uncertainty_mean','innovation_uncertainty'),
            ('retrieval_confidence_mean','retrieval_confidence'),
            ('predicted_utility_mean','predicted_utility'),
            ('predicted_risk_mean','predicted_risk'),
            ('innovation_abs','innovation_delta_abs'),
            ('retrieved_token_abs','retrieved_token_abs'),
        ):
            if field in output and isinstance(output[field],torch.Tensor):
                value=output[field]
                self._add(key,value.abs() if key.endswith('_abs') else value)
        if 'final_gate' in output:
            gate = output['final_gate'].detach()
            max_alpha = float(output.get('max_alpha', .6))
            self._add('gate_mean',gate)
            self._add('gate_near_zero_fraction',(gate < .01).float())
            self._add('gate_near_max_fraction',(gate >= max_alpha - .01).float())
        if 'base_prediction' not in output or 'candidate_prediction' not in output or 'prediction' not in output:
            return
        truth=batch['y_raw'].detach().float()
        valid=batch['future_valid'].detach().bool()
        base=(output['base_prediction'].detach().float()*std+mean-truth).abs()
        candidate=(output['candidate_prediction'].detach().float()*std+mean-truth).abs()
        final=(output['prediction'].detach().float()*std+mean-truth).abs()
        valid_float=valid.float()
        direct=valid & batch['direct_mask'].detach().bool()[:,None,:]
        indirect=valid & ~batch['direct_mask'].detach().bool()[:,None,:]
        for suffix, selection in (('',valid_float),('_direct',direct.float()),('_other',indirect.float())):
            self._add('base_MAE'+suffix,base,selection)
            self._add('candidate_MAE'+suffix,candidate,selection)
            self._add('final_MAE'+suffix,final,selection)
            self._add('candidate_MAE_gain'+suffix,base-candidate,selection)
            self._add('final_MAE_gain'+suffix,base-final,selection)
            self._add('candidate_helpful_fraction'+suffix,(candidate<base).float(),selection)
            self._add('candidate_harmful_fraction'+suffix,(candidate>base).float(),selection)
            self._add('final_helpful_fraction'+suffix,(final<base).float(),selection)
            self._add('final_harmful_fraction'+suffix,(final>base).float(),selection)
        self._add('candidate_correction_abs',(output['candidate_prediction']-output['base_prediction']).detach().abs())
        if 'final_gate' in output:
            activated=valid & (output['final_gate'].detach()>.05)
            self._add('gate_activated_fraction',activated.float(),valid_float)
            self._add('correction_harm_given_gate',(final>base).float(),activated.float())
            self._add('correction_gain_given_gate',base-final,activated.float())
        if 'predicted_utility' in output and output['predicted_utility'].shape==candidate.shape:
            predicted_positive=valid & (output['predicted_utility'].detach()>0)
            self._add('utility_predicted_positive_fraction',predicted_positive.float(),valid_float)
            self._add('utility_positive_candidate_precision',(candidate<base).float(),predicted_positive.float())

    def summary(self) -> dict[str, Any]:
        if not self.sampled_batches:
            return {'sampled_batches':0,'sampled_examples':0}
        # Single synchronisation per tensor at epoch end. No per-batch .item().
        result: dict[str, Any] = {'sampled_batches':self.sampled_batches,
                                  'sampled_examples':self.sampled_examples}
        for name,(numerator,denominator) in self.values.items():
            values=torch.stack([numerator.detach(),denominator.detach()]).cpu().tolist()
            result[name]=_safe_div(values[0],values[1])
        result['prediction_label_use']='monitoring_only_not_forward'
        return result


def append_diagnostics(run_dir: str | Path, phase: str, epoch: int,
                       train: dict[str, Any], val: dict[str, Any],
                       extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Append restart-safe JSONL and CSV, and return an stdout-friendly record."""
    root=Path(run_dir)/'diagnostics'
    root.mkdir(parents=True,exist_ok=True)
    record={'phase':phase,'epoch':int(epoch),'train':train,'val':val,'resource':extra or {}}
    with (root/'epoch_diagnostics.jsonl').open('a',encoding='utf-8') as fh:
        fh.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
    row={'phase':phase,'epoch':int(epoch)}
    for split,metrics in (('train',train),('val',val),('resource',extra or {})):
        for key,value in metrics.items():
            if isinstance(value,(int,float,str)) or value is None:
                row[f'{split}_{key}']=value
    csv_path=root/'epoch_diagnostics.csv'
    # Columns can evolve between stages, so rebuild cheaply once per epoch.
    rows=[]
    if csv_path.exists():
        with csv_path.open(encoding='utf-8',newline='') as fh:
            rows=list(csv.DictReader(fh))
    rows.append(row)
    fields=list(dict.fromkeys(key for item in rows for key in item))
    with csv_path.open('w',encoding='utf-8',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)
    return record
