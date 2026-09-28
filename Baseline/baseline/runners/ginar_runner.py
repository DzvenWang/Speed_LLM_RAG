#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# Keep the official GinAR numerical path in FP32.  The formal-run instability
# is handled at the optimizer/gradient-norm level rather than by changing the
# model precision.  stable_clip_grad_norm_ below accumulates norms in float64
# after scale-normalization, so very large but finite FP32 gradients do not make
# the global norm overflow before clipping can act.
GINAR_DTYPE = torch.float32
GINAR_DTYPE_NAME = 'float32'

ROOT = next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd())
sys.path.insert(0, str(ROOT))

from baseline.runners.common import (  # noqa: E402
    GinARDataset,
    make_run_dir,
    masked_l1,
    require_gpu,
    save_val_outputs,
    scale_stats,
    set_seed,
    validate_stage,
)


def transition_matrix(a: np.ndarray) -> np.ndarray:
    """Equivalent to official calculate_transition_matrix(adj)."""
    a = np.asarray(a, dtype=np.float32)
    rowsum = a.sum(axis=1)
    inv = np.zeros_like(rowsum, dtype=np.float32)
    nz = rowsum > 0
    inv[nz] = 1.0 / rowsum[nz]
    return (inv[:, None] * a).astype(np.float32)


def official_doubletransition(adj: np.ndarray, device: torch.device) -> list[torch.Tensor]:
    # GinAR official data_solve.load_adj(..., "doubletransition"):
    # calculate_transition_matrix(adj).T and calculate_transition_matrix(adj.T).T
    a1 = transition_matrix(adj).T
    a2 = transition_matrix(adj.T).T
    return [
        torch.from_numpy(a1).to(device=device, dtype=GINAR_DTYPE),
        torch.from_numpy(a2).to(device=device, dtype=GINAR_DTYPE),
    ]


def import_model():
    vendor = ROOT / 'vendor' / 'GinAR'
    sys.path.insert(0, str(vendor))
    try:
        from model1.ginar_arch import GinAR
    except Exception as e:
        raise RuntimeError(f'GinAR official import failed from {vendor}: {e!r}') from e
    return GinAR


class UpdateMilestoneScheduler:
    """Decay GinAR LR by optimizer-update count instead of enlarged Formal epochs.

    Development hyperparameters were selected with a fixed 12k training set.
    When Formal expands the train set, keeping epoch-number milestones would
    multiply the number of optimizer updates spent at each learning rate.
    This scheduler preserves the Development update budget represented by the
    configured epoch milestones, while leaving Development behavior unchanged.
    """

    def __init__(self, optimizer, milestone_steps, gamma: float):
        self.optimizer = optimizer
        self.milestone_steps = set(int(x) for x in milestone_steps)
        self.gamma = float(gamma)
        self.global_step = 0

    def step(self):
        self.global_step += 1
        if self.global_step in self.milestone_steps:
            for pg in self.optimizer.param_groups:
                pg['lr'] *= self.gamma
                pg['weight_decay'] = float(pg.get('weight_decay', 0.0)) * self.gamma
            print(
                f'GINAR LR UPDATE global_step={self.global_step} '
                f'lr={self.optimizer.param_groups[0]["lr"]:.8g}',
                flush=True,
            )


def tensor_stats(x: torch.Tensor) -> dict:
    """Compact finite-value diagnostics without moving full tensors to CPU."""
    x = x.detach()
    finite = torch.isfinite(x)
    finite_count = int(finite.sum().item())
    numel = int(x.numel())
    out = {
        'shape': list(x.shape),
        'dtype': str(x.dtype),
        'numel': numel,
        'finite_ratio': float(finite_count / max(numel, 1)),
        'nan_count': int(torch.isnan(x).sum().item()) if x.is_floating_point() else 0,
        'inf_count': int(torch.isinf(x).sum().item()) if x.is_floating_point() else 0,
    }
    if finite_count:
        xf = x[finite]
        xf64 = xf.to(dtype=torch.float64)
        out.update({
            'min': float(xf64.min().item()),
            'max': float(xf64.max().item()),
            'mean': float(xf64.mean().item()),
            'std': float(xf64.std(unbiased=False).item()),
            'max_abs': float(xf64.abs().max().item()),
        })
    else:
        out.update({'min': None, 'max': None, 'mean': None, 'std': None, 'max_abs': None})
    return out


def _index_list(idx) -> list[int]:
    if torch.is_tensor(idx):
        return [int(x) for x in idx.detach().cpu().reshape(-1).tolist()]
    if isinstance(idx, np.ndarray):
        return [int(x) for x in idx.reshape(-1).tolist()]
    if isinstance(idx, (list, tuple)):
        return [int(x) for x in idx]
    return [int(idx)]


def stable_clip_grad_norm_(
    model: nn.Module,
    max_norm: float,
    top_k: int = 8,
) -> tuple[float, list[dict], list[dict]]:
    """Robustly inspect and clip gradients without FP32 global-norm overflow.

    `torch.nn.utils.clip_grad_norm_` can report an infinite total norm even when
    every gradient element is finite, because squaring/accumulating very large
    FP32 values can overflow.  Here each parameter is first scale-normalized,
    its squared norm is accumulated in float64, and parameter norms are then
    combined with Python's stable `math.hypot`.

    Returns
    -------
    total_norm:
        Pre-clipping global L2 norm. NaN only when actual NaN/Inf gradient
        elements are present.
    nonfinite:
        Per-parameter diagnostics for real NaN/Inf gradient elements.
    largest:
        Largest finite per-parameter gradient norms for debugging.
    """
    finite_params: list[tuple[torch.nn.Parameter, float, str, float]] = []
    nonfinite: list[dict] = []

    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        finite = torch.isfinite(g)
        if not bool(finite.all()):
            gf = g[finite]
            nonfinite.append({
                'name': name,
                'shape': list(g.shape),
                'numel': int(g.numel()),
                'nan_count': int(torch.isnan(g).sum().item()),
                'inf_count': int(torch.isinf(g).sum().item()),
                'finite_max_abs': float(gf.abs().max().item()) if gf.numel() else None,
            })
            continue

        if g.numel() == 0:
            continue
        max_abs_t = g.abs().max()
        max_abs = float(max_abs_t.item())
        if max_abs == 0.0:
            param_norm = 0.0
        else:
            # Normalize before squaring so that the square cannot overflow.
            scaled = g / max_abs_t
            sq = torch.sum(scaled * scaled, dtype=torch.float64)
            param_norm = max_abs * math.sqrt(float(sq.item()))
        finite_params.append((p, param_norm, name, max_abs))

    if nonfinite:
        return float('nan'), nonfinite, []

    total_norm = 0.0
    for _p, param_norm, _name, _max_abs in finite_params:
        total_norm = math.hypot(total_norm, param_norm)
    if not math.isfinite(total_norm):
        raise FloatingPointError(
            f'GinAR stable global gradient norm is non-finite despite finite elements: {total_norm}'
        )

    largest = sorted(
        (
            {
                'name': name,
                'grad_norm': float(param_norm),
                'grad_max_abs': float(max_abs),
            }
            for _p, param_norm, name, max_abs in finite_params
        ),
        key=lambda z: z['grad_norm'],
        reverse=True,
    )[:max(int(top_k), 0)]

    if max_norm > 0.0 and total_norm > max_norm:
        scale = float(max_norm) / (total_norm + 1e-12)
        for p, _param_norm, _name, _max_abs in finite_params:
            p.grad.mul_(scale)

    return float(total_norm), [], largest


def _numerical_failure_payload(
    *, epoch: int, batch_id: int, idx, optimizer, step_scheduler,
    feat: torch.Tensor, y: torch.Tensor, pred: torch.Tensor,
    valid: torch.Tensor, loss: torch.Tensor, failure_type: str,
    gradient_failures=None, largest_gradients=None,
) -> dict:
    completed = int(step_scheduler.global_step) if step_scheduler is not None else None
    return {
        'failure_type': failure_type,
        'epoch': int(epoch),
        'batch_id': int(batch_id),
        'sample_indices': _index_list(idx),
        'optimizer_updates_completed': completed,
        'attempted_update': None if completed is None else completed + 1,
        'lr': float(optimizer.param_groups[0]['lr']) if optimizer is not None else None,
        'feat': tensor_stats(feat),
        'target': tensor_stats(y),
        'prediction': tensor_stats(pred),
        'valid_ratio': float(valid.float().mean().item()),
        'valid_count': int(valid.sum().item()),
        'loss': float(loss.detach().item()) if torch.isfinite(loss) else str(loss.detach().item()),
        'gradient_failures': gradient_failures or [],
        'largest_finite_gradients': largest_gradients or [],
    }


def run_epoch(
    model, loader, device, optimizer=None, clip=0.0, step_scheduler=None,
    epoch: int = 0, diagnostic_nonfinite: bool = False,
    diagnostic_max_failures: int = 8, max_optimizer_updates: int | None = None,
) -> dict:
    train = optimizer is not None
    model.train(train)
    total = 0.0
    denom = 0
    grad_norm_sum = 0.0
    grad_norm_max = 0.0
    grad_norm_count = 0
    numerical_failures = 0
    ctx = torch.enable_grad() if train else torch.no_grad()

    update_budget_exhausted = False
    with ctx:
        for batch_id, (feat, y, valid, idx) in enumerate(loader):
            if (
                train and step_scheduler is not None and max_optimizer_updates is not None
                and step_scheduler.global_step >= int(max_optimizer_updates)
            ):
                update_budget_exhausted = True
                break
            feat = feat.to(device=device, dtype=GINAR_DTYPE, non_blocking=True)
            y = y.to(device=device, dtype=GINAR_DTYPE, non_blocking=True)
            valid = valid.bool().to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)

            pred = model(feat)
            if pred.shape != y.shape:
                raise RuntimeError(
                    f'GinAR output contract failed: pred={tuple(pred.shape)} y={tuple(y.shape)}'
                )
            loss = masked_l1(pred, y, valid)

            if not torch.isfinite(loss):
                payload = _numerical_failure_payload(
                    epoch=epoch, batch_id=batch_id, idx=idx,
                    optimizer=optimizer, step_scheduler=step_scheduler,
                    feat=feat, y=y, pred=pred, valid=valid, loss=loss,
                    failure_type='nonfinite_loss',
                )
                print('GINAR NUMERICAL FAILURE ' + json.dumps(payload, default=str), flush=True)
                numerical_failures += 1
                if not diagnostic_nonfinite:
                    raise FloatingPointError('GinAR non-finite loss detected')
                if train:
                    optimizer.zero_grad(set_to_none=True)
                if numerical_failures >= diagnostic_max_failures:
                    raise FloatingPointError(
                        f'GinAR diagnostic reached max numerical failures={diagnostic_max_failures}'
                    )
                continue

            if train:
                loss.backward()
                grad_norm, nonfinite_grads, largest_grads = stable_clip_grad_norm_(
                    model, max_norm=clip, top_k=8,
                )
                if nonfinite_grads:
                    payload = _numerical_failure_payload(
                        epoch=epoch, batch_id=batch_id, idx=idx,
                        optimizer=optimizer, step_scheduler=step_scheduler,
                        feat=feat, y=y, pred=pred, valid=valid, loss=loss,
                        failure_type='actual_nonfinite_gradient_elements',
                        gradient_failures=nonfinite_grads,
                        largest_gradients=largest_grads,
                    )
                    print('GINAR NUMERICAL FAILURE ' + json.dumps(payload, default=str), flush=True)
                    numerical_failures += 1
                    optimizer.zero_grad(set_to_none=True)
                    if not diagnostic_nonfinite:
                        raise FloatingPointError(
                            'GinAR actual NaN/Inf gradient elements detected'
                        )
                    if numerical_failures >= diagnostic_max_failures:
                        raise FloatingPointError(
                            f'GinAR diagnostic reached max numerical failures={diagnostic_max_failures}'
                        )
                    continue

                grad_norm_sum += float(grad_norm)
                grad_norm_max = max(grad_norm_max, float(grad_norm))
                grad_norm_count += 1

                optimizer.step()

                # Catch optimizer-state/parameter corruption immediately rather
                # than allowing it to surface as a later forward NaN.
                bad_params = []
                for name, p in model.named_parameters():
                    if not bool(torch.isfinite(p.detach()).all()):
                        bad_params.append(name)
                if bad_params:
                    payload = _numerical_failure_payload(
                        epoch=epoch, batch_id=batch_id, idx=idx,
                        optimizer=optimizer, step_scheduler=step_scheduler,
                        feat=feat, y=y, pred=pred, valid=valid, loss=loss,
                        failure_type='nonfinite_parameter_after_optimizer_step',
                        gradient_failures=[{'name': n} for n in bad_params[:32]],
                        largest_gradients=largest_grads,
                    )
                    print('GINAR NUMERICAL FAILURE ' + json.dumps(payload, default=str), flush=True)
                    raise FloatingPointError(
                        'GinAR parameter became NaN/Inf after optimizer.step()'
                    )

                if step_scheduler is not None:
                    step_scheduler.step()

            n = int(valid.sum().item())
            total += float(loss.detach().item()) * max(n, 1)
            denom += max(n, 1)

    return {
        'mae': total / max(denom, 1),
        'grad_norm_mean': (
            grad_norm_sum / grad_norm_count if grad_norm_count else None
        ),
        'grad_norm_max': grad_norm_max if grad_norm_count else None,
        'grad_norm_count': int(grad_norm_count),
        'numerical_failures': int(numerical_failures),
        'optimizer_updates_completed': (
            int(step_scheduler.global_step) if step_scheduler is not None else None
        ),
        'update_budget_exhausted': bool(update_budget_exhausted),
    }


def predict(model, loader, device, mean: float, std: float) -> np.ndarray:
    out = []
    model.eval()
    with torch.no_grad():
        for feat, _y, _valid, _idx in loader:
            pred = model(feat.to(device=device, dtype=GINAR_DTYPE, non_blocking=True))
            out.append(pred.detach().cpu().numpy())
    pred_norm = np.concatenate(out, axis=0).astype(np.float32)
    return pred_norm * np.float32(std) + np.float32(mean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='development', choices=['development', 'formal'])
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--smoke', action='store_true', help='Run one train and one val batch only; do not save a formal result.')
    ap.add_argument('--diagnostic-nonfinite', action='store_true', help='Diagnostic only: log and skip NaN/Inf batches; never save a formal result.')
    ap.add_argument('--stability-probe', action='store_true', help='Use the full Formal cache for a short stability check and never save a selectable result.')
    ap.add_argument('--diagnostic-max-failures', type=int, default=8, help='Maximum numerical failures to collect in diagnostic mode before aborting.')
    args = ap.parse_args()

    cfg, cache = validate_stage(args.stage, 'GinAR')
    rcfg = cfg['runners']['GinAR']
    seed = int(cfg['experiment']['seed'] if args.seed is None else args.seed)
    set_seed(seed)
    device = require_gpu()

    train_ds = GinARDataset(cache, 'train')
    val_ds = GinARDataset(cache, 'val')
    if train_ds.x.shape[1:] != (int(cfg['experiment']['history']), 388):
        raise RuntimeError(f'Unexpected GinAR train input shape {train_ds.x.shape}')

    batch = int(rcfg['batch_size'])
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    GinAR = import_model()
    adj = np.load(cache / 'adjacency.npy')
    supports = official_doubletransition(adj, device)
    model = GinAR(
        int(cfg['experiment']['history']),
        int(adj.shape[0]),
        int(cfg['experiment']['horizon']),
        3,
        int(rcfg['emb_size']),
        int(rcfg['graph_size']),
        int(rcfg['layers']),
        float(rcfg['dropout']),
        supports,
    ).to(device=device, dtype=GINAR_DTYPE)

    # graph_data is a plain Python list in the official GinAR implementation,
    # not a registered buffer, so model.to(dtype=...) does not convert it.
    # Keep both parameters and graph supports on the official FP32 path.
    model.graph_data = [
        g.to(device=device, dtype=GINAR_DTYPE) for g in model.graph_data
    ]
    bad_param_dtypes = [
        (name, str(p.dtype))
        for name, p in model.named_parameters()
        if p.dtype != GINAR_DTYPE
    ]
    bad_support_dtypes = [
        (i, str(g.dtype))
        for i, g in enumerate(model.graph_data)
        if g.dtype != GINAR_DTYPE
    ]
    if bad_param_dtypes or bad_support_dtypes:
        raise RuntimeError(
            'GinAR FP32 dtype contract failed: '
            f'params={bad_param_dtypes[:8]} supports={bad_support_dtypes}'
        )

    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f'GINAR START stage={args.stage} seed={seed} device={device} '
        f'params={nparams} dtype={GINAR_DTYPE_NAME}',
        flush=True,
    )
    print(
        'GINAR FP32 MODEL CONTRACT PASS '
        f'parameter_dtype={GINAR_DTYPE_NAME} support_dtype={GINAR_DTYPE_NAME}',
        flush=True,
    )
    print(f'GINAR DATA train={len(train_ds)} val={len(val_ds)} H={train_ds.x.shape[1]} N={train_ds.x.shape[2]}', flush=True)

    base_lr = float(rcfg['learning_rate'])
    milestones = set(int(x) for x in rcfg.get('lr_milestones', []))
    gamma = float(rcfg.get('lr_gamma', 0.5))
    sorted_milestones = sorted(milestones)

    # Keep the official GinAR optimization mechanism intact. Development uses
    # the configured official-style LR/schedule. Formal starts from a frozen
    # Development-selected stability LR, but it does not pretend that a random
    # initialization has already consumed earlier Development milestones.
    effective_lr = (
        float(rcfg.get('formal_learning_rate', base_lr))
        if args.stage == 'formal' else base_lr
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr)

    # Formal has ~4.7x more train rows than Development. Preserve the frozen
    # Development optimizer-update budget and epoch-equivalent *future* decay
    # locations instead of multiplying optimization simply because the data
    # population is larger. No network/loss/optimizer mechanism is changed.
    step_scheduler = None
    formal_milestone_steps = []
    reference_steps_per_epoch = None
    formal_update_budget = None
    formal_lr_milestones = []
    if args.stage == 'formal':
        reference_rows = int(cfg['protocol_assertions']['development']['train_rows'])
        reference_steps_per_epoch = (reference_rows + batch - 1) // batch
        formal_lr_milestones = [
            int(x) for x in rcfg.get('formal_lr_milestones', [40, 70, 90])
        ]
        if any(ep not in milestones for ep in formal_lr_milestones):
            raise RuntimeError(
                'GinAR formal_lr_milestones must be a subset of the frozen '
                f'Development schedule: formal={formal_lr_milestones}, development={sorted_milestones}'
            )
        formal_milestone_steps = [
            int(ep) * reference_steps_per_epoch for ep in formal_lr_milestones
        ]
        formal_update_budget = int(
            rcfg.get(
                'formal_max_optimizer_updates',
                int(rcfg['max_epochs']) * reference_steps_per_epoch,
            )
        )
        if formal_update_budget < 1:
            raise RuntimeError('GinAR formal_max_optimizer_updates must be positive')
        step_scheduler = UpdateMilestoneScheduler(
            optimizer, formal_milestone_steps, gamma
        )
        print(
            'GINAR FORMAL UPDATE-BUDGET POLICY '
            f'base_lr={base_lr:.8g} formal_lr={effective_lr:.8g} '
            f'reference_rows={reference_rows} batch={batch} '
            f'reference_steps_per_epoch={reference_steps_per_epoch} '
            f'formal_lr_milestones={formal_lr_milestones} '
            f'milestone_steps={formal_milestone_steps} '
            f'max_optimizer_updates={formal_update_budget} dtype={GINAR_DTYPE_NAME}',
            flush=True,
        )
    max_epochs = 1 if args.smoke else (int(rcfg.get('stability_probe_epochs', 8)) if args.stability_probe else int(rcfg['max_epochs']))
    patience = int(rcfg.get('patience', 20))
    clip = float(rcfg.get('gradient_clip', 5.0))

    best = float('inf'); best_state = None; best_epoch = 0; stale = 0
    termination_reason = 'max_epochs'
    guard_enabled = bool(rcfg.get('divergence_guard_enabled', True))
    guard_min_epoch = int(rcfg.get('divergence_guard_min_epoch', 2))
    guard_ratio = float(rcfg.get('divergence_guard_ratio', 5.0))
    guard_abs = float(rcfg.get('divergence_guard_abs', 5.0))
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        if args.smoke:
            # Minimal real official-model forward/backward contract.
            feat, y, valid, _ = next(iter(train_loader))
            optimizer.zero_grad(set_to_none=True)
            pred = model(feat.to(device=device, dtype=GINAR_DTYPE)); loss = masked_l1(pred, y.to(device=device, dtype=GINAR_DTYPE), valid.bool().to(device))
            loss.backward()
            smoke_grad_norm, smoke_bad_grads, _ = stable_clip_grad_norm_(model, clip)
            if smoke_bad_grads:
                raise FloatingPointError(
                    'GinAR smoke detected actual NaN/Inf gradient elements: ' +
                    json.dumps(smoke_bad_grads, default=str)
                )
            optimizer.step()
            vfeat, vy, vv, _ = next(iter(val_loader))
            model.eval()
            with torch.no_grad():
                vp = model(vfeat.to(device=device, dtype=GINAR_DTYPE)); vl = masked_l1(vp, vy.to(device=device, dtype=GINAR_DTYPE), vv.bool().to(device))
            print(f'GINAR GPU SMOKE PASS train_loss={loss.item():.6f} val_loss={vl.item():.6f} grad_norm={smoke_grad_norm:.6g} output={tuple(vp.shape)}', flush=True)
            return

        et = time.time()
        tr = run_epoch(
            model, train_loader, device, optimizer=optimizer, clip=clip,
            step_scheduler=step_scheduler, epoch=epoch,
            diagnostic_nonfinite=args.diagnostic_nonfinite,
            diagnostic_max_failures=int(args.diagnostic_max_failures),
            max_optimizer_updates=formal_update_budget,
        )
        va = run_epoch(
            model, val_loader, device, optimizer=None, epoch=epoch,
            diagnostic_nonfinite=False,
        )
        if args.stage != 'formal' and epoch in milestones:
            for pg in optimizer.param_groups:
                pg['lr'] *= gamma
                pg['weight_decay'] = float(pg.get('weight_decay', 0.0)) * gamma
        lr = optimizer.param_groups[0]['lr']
        print(
            f'EPOCH {epoch:03d} train_MAE_norm={tr["mae"]:.6f} '
            f'val_MAE_norm={va["mae"]:.6f} lr={lr:.8g} '
            f'grad_norm_mean={tr["grad_norm_mean"]} '
            f'grad_norm_max={tr["grad_norm_max"]} '
            f'numerical_failures={tr["numerical_failures"]} '
            f'sec={time.time()-et:.1f}',
            flush=True,
        )
        if args.diagnostic_nonfinite:
            # Diagnostic mode intentionally cannot create/select a formal checkpoint.
            continue

        # Numerical safeguard only: when an already-finite best checkpoint exists
        # and validation catastrophically explodes, stop before a later optimizer
        # update can corrupt the run. This does not skip/repair any bad batch.
        if (
            args.stage == 'formal' and guard_enabled and best_state is not None
            and epoch >= guard_min_epoch
        ):
            ratio = float(va['mae']) / max(float(best), 1e-12)
            if float(va['mae']) > guard_abs or ratio > guard_ratio:
                termination_reason = 'validation_divergence'
                msg = (
                    'GINAR VALIDATION DIVERGENCE STOP '
                    f'epoch={epoch} val={va["mae"]:.6f} best={best:.6f} '
                    f'ratio={ratio:.3f} abs_guard={guard_abs:.6g} ratio_guard={guard_ratio:.6g}'
                )
                print(msg, flush=True)
                # Treat catastrophic post-best validation divergence as a guarded
                # early-stop.  The best checkpoint was already selected using Val-only
                # criteria before the divergence, so preserve that finite checkpoint and
                # stop further optimizer updates instead of rejecting the whole run.
                # Stability-probe mode remains fail-fast because it is explicitly meant
                # to diagnose whether the training trajectory itself stays stable.
                if args.stability_probe:
                    raise RuntimeError(msg + ' -- stability probe rejected')
                print(
                    'GINAR DIVERGENCE GUARD EARLY-STOP: preserving previously selected '
                    f'best_epoch={best_epoch} best_val_MAE_norm={best:.6f}',
                    flush=True,
                )
                break

        if va["mae"] < best - 1e-8:
            best = va["mae"]; best_epoch = epoch; stale = 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
            if stale >= patience:
                termination_reason = 'early_stopping'
                print(f'EARLY STOP epoch={epoch} best_epoch={best_epoch}', flush=True)
                break

        if tr.get('update_budget_exhausted', False):
            termination_reason = 'optimizer_update_budget'
            print(
                f'GINAR UPDATE BUDGET STOP epoch={epoch} '
                f'updates={tr.get("optimizer_updates_completed")} budget={formal_update_budget}',
                flush=True,
            )
            break

    if args.diagnostic_nonfinite:
        print(
            'GINAR DIAGNOSTIC COMPLETE -- NO FORMAL RESULT SAVED',
            flush=True,
        )
        return

    if args.stability_probe:
        print(
            f'GINAR STABILITY PROBE PASS epochs={max_epochs} best_val_MAE_norm={best:.6f}',
            flush=True,
        )
        return

    if best_state is None:
        raise RuntimeError('GinAR did not produce a checkpoint')
    model.load_state_dict(best_state)
    model.to(device=device, dtype=GINAR_DTYPE)
    model.graph_data = [
        g.to(device=device, dtype=GINAR_DTYPE) for g in model.graph_data
    ]
    mean, std = scale_stats(cache)
    pred_raw = predict(model, val_loader, device, mean, std)
    run_dir = make_run_dir(cfg, 'GinAR', args.stage, seed)
    torch.save({
        'model': best_state,
        'best_epoch': best_epoch,
        'best_val_mae_norm': best,
        'termination_reason': termination_reason,
        'formal_max_optimizer_updates': formal_update_budget,
    }, run_dir / 'best.pt')
    summary = save_val_outputs(cfg, cache, run_dir, pred_raw, 'GinAR', seed, args.stage, {
        'best_epoch': int(best_epoch),
        'best_val_MAE_norm': float(best),
        'params': int(nparams),
        'wall_seconds': float(time.time() - t0),
        'official_architecture': 'vendor/GinAR/model1/ginar_arch.py::GinAR',
        'graph_support': 'official_doubletransition',
        'checkpoint_selection': 'val_only',
        'lr_schedule_mode': (
            'development_epoch_milestones' if args.stage != 'formal'
            else 'formal_development_update_budget_with_frozen_stability_lr'
        ),
        'base_learning_rate': float(base_lr),
        'effective_initial_learning_rate': float(effective_lr),
        'formal_lr_milestones': formal_lr_milestones,
        'formal_reference_steps_per_epoch': reference_steps_per_epoch,
        'formal_lr_milestone_steps': formal_milestone_steps,
        'formal_max_optimizer_updates': formal_update_budget,
        'termination_reason': termination_reason,
        'divergence_guard': {
            'enabled': guard_enabled, 'min_epoch': guard_min_epoch,
            'ratio_to_best': guard_ratio, 'absolute_ceiling': guard_abs,
        },
        'nonfinite_guard': True,
        'gradient_norm_mode': 'scaled_float64_accumulation',
        'gradient_element_finite_check': True,
        'post_optimizer_parameter_finite_check': True,
        'numerical_precision': GINAR_DTYPE_NAME,
        'model_precision_policy': 'official_fp32',
        'gradient_norm_accumulation_precision': 'float64',
        'formal_stability_policy': 'conservative_formal_lr+clip1+development_update_budget+guarded_early_stop_on_post_best_divergence+fail_fast_nonfinite',
        'divergence_guard_triggered': termination_reason == 'validation_divergence',
        'selection_valid': True,
        'test_access': False,
    })
    print('GINAR COMPLETE', json.dumps({'run_dir': str(run_dir), 'val': summary['val']}, default=str), flush=True)


if __name__ == '__main__':
    main()
