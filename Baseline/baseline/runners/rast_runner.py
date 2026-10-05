#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import copy
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd())
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from baseline.runners.common import (  # noqa: E402
    RASTDataset,
    make_run_dir,
    masked_l1,
    require_gpu,
    save_val_outputs,
    scale_stats,
    set_seed,
    validate_stage,
)
from baseline.backbones.reference_backbone import (  # noqa: E402
    build_reference_backbone,
    ReferenceBackboneRASTBridge,
    SUPPORTED_BACKBONES,
)
from baseline.compat.rast_optional_backbones import install_rast_native_import_shims  # noqa: E402
from baseline.runners.retrieval_impact import (  # noqa: E402
    accumulate_impact,
    empty_impact_stats,
    finalize_impact,
)


def import_model():
    vendor = ROOT / 'vendor' / 'RAST'
    sys.path.insert(0, str(vendor))
    modes = install_rast_native_import_shims()
    print(f'RAST NATIVE IMPORT SHIMS={modes}', flush=True)
    try:
        from src.arch.rast_arch import RAST
    except Exception as e:
        raise RuntimeError(
            'RAST import failed after FAISS/native-import compatibility setup. '
            + repr(e)
        ) from e
    return RAST


def build_model(cfg, variant, seed, device, db, backbone_model=None, backbone_checkpoint=None):
    c = cfg['runners']['RAST']
    n_nodes = 388
    RAST = import_model()
    model = RAST(
        num_nodes=n_nodes,
        input_dim=3,
        output_dim=1,
        input_len=int(cfg['experiment']['history']),
        output_len=int(cfg['experiment']['horizon']),
        dropout=float(c['dropout']),
        query_dim=int(c['query_dim']),
        retrieval_dim=int(c['retrieval_dim']),
        temporal_dim=int(c['temporal_dim']),
        spatial_dim=int(c['spatial_dim']),
        update_interval=int(c['update_interval']),
        encoder_layers=int(c['encoder_layers']),
        top_k=int(c['top_k']),
        database_path=str(db),
        use_amp=bool(c['use_amp']),
    ).to(device)
    if variant in ('patchstg', 'reference'):
        selected = 'PatchSTG' if variant == 'patchstg' else str(backbone_model or '')
        if not selected:
            raise RuntimeError('RAST reference variant requires --backbone')
        reference = build_reference_backbone(
            cfg, device, model=selected, seed=int(cfg.get('reference_backbone',{}).get('seed', seed)),
            checkpoint=backbone_checkpoint,
        )
        model.backbone = ReferenceBackboneRASTBridge(reference)
        model._reference_backbone_model = selected
        model._reference_backbone_checkpoint = reference.checkpoint_reference
        model._reference_backbone_checkpoint_sha256 = str(reference.checkpoint_sha256)
        model._reference_backbone_seed = int(reference.reference_seed)
        safe_residual = bool(c.get('reference_control_zero_init_residual_scale', True))
        model._safe_residual = safe_residual
        model._residual_scale_init = float(c.get('reference_control_residual_scale_init', 0.0))
        if safe_residual:
            # The RAST model has already been moved to ``device`` above, so create the
            # compatibility scalar directly on that device. Native RAST is untouched.
            model.reference_residual_scale = nn.Parameter(
                torch.tensor(model._residual_scale_init, dtype=torch.float32, device=device)
            )
    return model


def _residual_scale_value(model) -> float | None:
    alpha = getattr(model, 'reference_residual_scale', None)
    if alpha is None:
        return None
    return float(alpha.detach().cpu().item())


def _apply_matched_safe_residual(model, pred: torch.Tensor) -> torch.Tensor:
    """Calibrate transplanted RAST fusion against the selected reference Backbone.

    Matched-backbone variants use ``base + alpha * (rast_fused - base)`` with
    zero-initialized learnable ``alpha``. Native RAST never enters this path.
    """
    if not bool(getattr(model, '_safe_residual', False)):
        return pred
    bridge = getattr(model, 'backbone', None)
    base = getattr(bridge, 'last_prediction', None)
    if base is None:
        raise RuntimeError('RAST safe residual requested but reference Backbone prediction is unavailable')
    if tuple(base.shape) != tuple(pred.shape):
        raise RuntimeError(
            f'RAST safe residual shape mismatch base={tuple(base.shape)} pred={tuple(pred.shape)}'
        )
    alpha = getattr(model, 'reference_residual_scale', None)
    if alpha is None:
        raise RuntimeError('RAST safe residual enabled without reference_residual_scale parameter')
    correction = pred - base
    return base + alpha.to(dtype=correction.dtype) * correction


def snapshot_store(model):
    r = model.retrieval_store
    return {
        'tv': np.asarray(list(r.temporal_vectors), np.float32),
        'sv': np.asarray(list(r.spatial_vectors), np.float32),
        'tt': [np.asarray(x, np.float32) for x in getattr(r, 'temporal_tensors', [])],
        'st': [np.asarray(x, np.float32) for x in getattr(r, 'spatial_tensors', [])],
    }


def restore_store(model, snap):
    r = model.retrieval_store
    r.temporal_vectors = deque([x for x in snap['tv']], maxlen=r.expected_size)
    r.spatial_vectors = deque([x for x in snap['sv']], maxlen=r.expected_size)
    r.temporal_tensors = list(snap.get('tt', []))
    r.spatial_tensors = list(snap.get('st', []))
    r._rebuild_indices()


def retrieval_store_status(model):
    r = model.retrieval_store
    status = {
        'temporal_vectors': len(getattr(r, 'temporal_vectors', [])),
        'spatial_vectors': len(getattr(r, 'spatial_vectors', [])),
        'temporal_tensors': len(getattr(r, 'temporal_tensors', [])),
        'spatial_tensors': len(getattr(r, 'spatial_tensors', [])),
    }
    status['ready'] = all(
        status[k] > 0
        for k in ('temporal_vectors', 'spatial_vectors', 'temporal_tensors', 'spatial_tensors')
    )
    return status


def save_store(path, snap):
    kw = {
        'temporal_vectors': snap['tv'],
        'spatial_vectors': snap['sv'],
        'nt': np.int64(len(snap['tt'])),
        'ns': np.int64(len(snap['st'])),
    }
    for i, x in enumerate(snap['tt']):
        kw[f'tt_{i}'] = x
    for i, x in enumerate(snap['st']):
        kw[f'st_{i}'] = x
    np.savez_compressed(path, **kw)


def load_store(path):
    with np.load(path, allow_pickle=False) as z:
        nt = int(z['nt'])
        ns = int(z['ns'])
        return {
            'tv': z['temporal_vectors'],
            'sv': z['spatial_vectors'],
            'tt': [z[f'tt_{i}'] for i in range(nt)],
            'st': [z[f'st_{i}'] for i in range(ns)],
        }


def _correction_stats(model, pred: torch.Tensor, valid: torch.Tensor) -> tuple[float, float, float, int] | None:
    """Measure how much RAST changes the selected reference-Backbone prediction."""
    bridge = getattr(model, 'backbone', None)
    base = getattr(bridge, 'last_prediction', None)
    if base is None:
        return None
    if tuple(base.shape) != tuple(pred.shape):
        raise RuntimeError(
            f'RAST-reference correction contract failed: base={tuple(base.shape)} pred={tuple(pred.shape)}'
        )
    corr = (pred.detach() - base.detach())
    mask = valid.bool()
    if not torch.any(mask):
        vals = corr.reshape(-1)
    else:
        vals = corr[mask]
    if vals.numel() == 0:
        return 0.0, 0.0, 0.0, 0
    abs_sum = float(vals.abs().sum().item())
    sq_sum = float((vals.double() * vals.double()).sum().item())
    max_abs = float(vals.abs().max().item())
    return abs_sum, sq_sum, max_abs, int(vals.numel())


def run_epoch(model, dl, device, ep, train, opt=None, max_batches=None, clip=5.0):
    model.train(train)
    total = 0.0
    denom = 0
    corr_abs_sum = 0.0
    corr_sq_sum = 0.0
    corr_max_abs = 0.0
    corr_count = 0
    impact_stats = empty_impact_stats()
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_i, (idx, x, y, valid) in enumerate(dl):
            x = x.float().to(device)
            y = y.float().to(device)
            valid = valid.bool().to(device)
            if train:
                if opt is None:
                    raise RuntimeError('RAST training epoch requires optimizer')
                opt.zero_grad(set_to_none=True)
            out = model(x, None, int(idx[0]) if len(idx) else 0, ep, train)
            pred = out['prediction'].squeeze(-1)
            if getattr(model, '_safe_residual', False):
                pred = _apply_matched_safe_residual(model, pred)
            if tuple(pred.shape) != tuple(y.shape):
                raise RuntimeError(f'RAST output shape {tuple(pred.shape)} != target {tuple(y.shape)}')
            if not torch.isfinite(pred).all():
                raise FloatingPointError(f'RAST produced non-finite prediction at epoch={ep} batch={batch_i}')
            loss = masked_l1(pred, y, valid)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'RAST produced non-finite loss at epoch={ep} batch={batch_i}')

            corr = _correction_stats(model, pred, valid)
            if corr is not None:
                a, s, m, n_corr = corr
                corr_abs_sum += a
                corr_sq_sum += s
                corr_max_abs = max(corr_max_abs, m)
                corr_count += n_corr

            bridge = getattr(model, 'backbone', None)
            base = getattr(bridge, 'last_prediction', None)
            if base is not None:
                accumulate_impact(impact_stats, pred, base, y, valid)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [q for q in model.parameters() if q.requires_grad],
                    float(clip),
                )
                opt.step()
            n = int(valid.sum())
            total += float(loss.item()) * max(n, 1)
            denom += max(n, 1)
            if max_batches is not None and batch_i + 1 >= int(max_batches):
                break

    result = {'mae': total / max(denom, 1)}
    if corr_count:
        result.update({
            'correction_abs_mean': corr_abs_sum / corr_count,
            'correction_rms': float(np.sqrt(corr_sq_sum / corr_count)),
            'correction_max_abs': corr_max_abs,
            'correction_count': corr_count,
        })
    else:
        result.update({
            'correction_abs_mean': None,
            'correction_rms': None,
            'correction_max_abs': None,
            'correction_count': 0,
        })
    result['impact'] = finalize_impact(impact_stats)
    return result


def prewarm_reference_retrieval(model, train_loader, device, logical_epoch: int) -> dict:
    """Populate RAST retrieval memory before the first optimizer update.

    Official RAST only populates the store when ``train=True`` and
    ``epoch % update_interval == 0``.  With a strong reference backbone,
    optimizing for nine empty-retrieval epochs drives the residual correction
    toward zero before retrieval becomes available.  This warm-up performs one
    train-data forward under no_grad, preserving the official store update code
    while making the first optimizer step retrieval-aware.
    """
    before = retrieval_store_status(model)
    model.train(True)
    idx, x, _y, _valid = next(iter(train_loader))
    x = x.float().to(device)
    with torch.no_grad():
        out = model(x, None, int(idx[0]) if len(idx) else 0, logical_epoch, True)
        pred = out['prediction'].squeeze(-1)
        if not torch.isfinite(pred).all():
            raise FloatingPointError('RAST-reference retrieval prewarm produced non-finite prediction')
    after = retrieval_store_status(model)
    if not after['ready']:
        raise RuntimeError(
            'RAST-reference retrieval-first warm-up failed to populate retrieval memory: '
            + json.dumps({'before': before, 'after': after}, sort_keys=True)
        )
    print(
        'RAST-REFERENCE RETRIEVAL PREWARM PASS '
        + json.dumps({'logical_epoch': int(logical_epoch), 'before': before, 'after': after}, sort_keys=True),
        flush=True,
    )
    return after


def predict(model, dl, device, mean, std, ep, collect_impact=False):
    out = []
    impact_stats = empty_impact_stats()
    model.eval()
    with torch.no_grad():
        for idx, x, y, valid in dl:
            x = x.float().to(device)
            y = y.float().to(device)
            valid = valid.bool().to(device)
            pred = model(
                x, None,
                int(idx[0]) if len(idx) else 0,
                ep, False,
            )['prediction'].squeeze(-1)
            if getattr(model, '_safe_residual', False):
                pred = _apply_matched_safe_residual(model, pred)
            if not torch.isfinite(pred).all():
                raise FloatingPointError('RAST produced non-finite validation prediction')
            if collect_impact:
                bridge = getattr(model, 'backbone', None)
                base = getattr(bridge, 'last_prediction', None)
                if base is None:
                    raise RuntimeError(
                        'RAST matched-backbone diagnostics requested but reference Backbone prediction is unavailable'
                    )
                accumulate_impact(impact_stats, pred, base, y, valid)
            out.append(pred.cpu().numpy())
    raw = np.concatenate(out, 0).astype(np.float32) * np.float32(std) + np.float32(mean)
    impact = finalize_impact(impact_stats) if collect_impact else None
    return raw, impact

def _save_lifecycle_smoke_checkpoint(payload, retrieval_store=None):
    out = os.environ.get("BASELINE_LIFECYCLE_CHECKPOINT")
    if not out:
        return None
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    if retrieval_store is not None:
        save_store(path.with_name("retrieval_store_best.npz"), retrieval_store)
    print(f"LIFECYCLE_CHECKPOINT_SAVE_PASS path={path}", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['development', 'formal'], default='formal')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--variant', choices=['native', 'patchstg', 'reference'], default='native')
    ap.add_argument('--backbone', choices=SUPPORTED_BACKBONES)
    ap.add_argument('--backbone-checkpoint', default=None)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    if args.variant in ('patchstg','reference') and args.stage != 'formal':
        raise SystemExit('RAST matched-backbone controlled adaptation is formal-only')
    if args.variant == 'reference' and not args.backbone:
        raise SystemExit('--backbone is required for --variant reference')

    cfg, cache = validate_stage(args.stage, 'RAST')
    set_seed(args.seed)
    device = require_gpu()
    c = cfg['runners']['RAST']
    selected_backbone = ('PatchSTG' if args.variant == 'patchstg' else (args.backbone if args.variant == 'reference' else None))
    label = 'RAST' if args.variant == 'native' else f'RAST-{selected_backbone}'

    train_ds = RASTDataset(cache, 'train')
    val_ds = RASTDataset(cache, 'val')
    loader = lambda d, shuffle: DataLoader(
        d,
        batch_size=int(c['batch_size']),
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )
    train_loader = loader(train_ds, True)
    val_loader = loader(val_ds, False)

    temp = Path(cfg['project']['exp_root']) / '_tmp_rast' / f'{label}_{args.seed}_{int(time.time())}'
    temp.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg, args.variant, args.seed, device, temp, backbone_model=selected_backbone, backbone_checkpoint=args.backbone_checkpoint)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(
        params,
        lr=float(c['learning_rate']),
        weight_decay=float(c['weight_decay']),
    )
    clip = float(c.get('gradient_clip', 5.0))

    retrieval_first = bool(c.get('patchstg_retrieval_first', True))
    prewarm_epoch = int(c['update_interval'])
    prewarm_status = None
    if args.variant != 'native' and retrieval_first:
        prewarm_status = prewarm_reference_retrieval(
            model, train_loader, device, prewarm_epoch,
        )

    if args.smoke:
        logical_epoch = prewarm_epoch if args.variant != 'native' else 1
        smoke = run_epoch(
            model, train_loader, device, logical_epoch, True, opt,
            max_batches=1, clip=clip,
        )
        if not np.isfinite(smoke['mae']):
            raise RuntimeError(f'{label} smoke returned non-finite loss: {smoke}')
        store_status = retrieval_store_status(model)
        if args.variant != 'native':
            if not store_status['ready']:
                raise RuntimeError(
                    'RAST matched-backbone smoke did not populate retrieval memory: '
                    + json.dumps(store_status, sort_keys=True)
                )
            print(
                'RAST-REFERENCE GPU SMOKE PASS',
                {
                    'loss': smoke['mae'],
                    'logical_epoch': logical_epoch,
                    'retrieval_store': store_status,
                    'correction_abs_mean': smoke['correction_abs_mean'],
                    'correction_rms': smoke['correction_rms'],
                    'correction_max_abs': smoke['correction_max_abs'],
                    'impact': smoke['impact'],
                    'reference_backbone_model': getattr(model, '_reference_backbone_model', None),
                    'reference_backbone_seed': getattr(model, '_reference_backbone_seed', None),
                    'reference_backbone_checkpoint': getattr(model, '_reference_backbone_checkpoint', None),
                    'safe_residual': bool(getattr(model, '_safe_residual', False)),
                    'residual_scale': _residual_scale_value(model),
                },
                flush=True,
            )
        else:
            print('RAST GPU SMOKE PASS', smoke['mae'], flush=True)
        _save_lifecycle_smoke_checkpoint({'model': {k:v.detach().cpu() for k,v in model.state_dict().items()}, 'best_epoch': logical_epoch, 'lifecycle_smoke': True}, snapshot_store(model))
        return

    best = float('inf')
    state = None
    store = None
    best_logical_epoch = 0
    best_train_epoch = 0
    best_val_impact = None
    stale = 0
    t0 = time.time()
    first_eligible_epoch = None
    min_delta = (
        float(c.get('reference_control_min_delta', 1.0e-8))
        if args.variant != 'native'
        else float(c.get('min_delta', 1.0e-8))
    )

    for train_epoch in range(1, int(c['max_epochs']) + 1):
        logical_epoch = (
            prewarm_epoch + train_epoch - 1
            if args.variant != 'native' and retrieval_first
            else train_epoch
        )
        train_result = run_epoch(
            model, train_loader, device, logical_epoch, True, opt, clip=clip,
        )
        val_result = run_epoch(
            model, val_loader, device, logical_epoch, False, None, clip=clip,
        )
        store_status = retrieval_store_status(model)
        checkpoint_eligible = (args.variant == 'native' or store_status['ready'])
        if checkpoint_eligible and first_eligible_epoch is None:
            first_eligible_epoch = logical_epoch

        print(
            f'EPOCH train={train_epoch:03d} logical={logical_epoch:03d} '
            f'train_MAE_norm={train_result["mae"]:.6f} '
            f'val_MAE_norm={val_result["mae"]:.6f} '
            f'checkpoint_eligible={checkpoint_eligible} '
            f'retrieval_ready={store_status["ready"]} '
            f'tv={store_status["temporal_vectors"]} sv={store_status["spatial_vectors"]} '
            f'tt={store_status["temporal_tensors"]} st={store_status["spatial_tensors"]} '
            f'train_corr_abs_mean={train_result["correction_abs_mean"]} '
            f'val_corr_abs_mean={val_result["correction_abs_mean"]} '
            f'val_corr_rms={val_result["correction_rms"]} '
            f'val_helpful={None if val_result["impact"] is None else val_result["impact"]["helpful_fraction"]} '
            f'val_harmful={None if val_result["impact"] is None else val_result["impact"]["harmful_fraction"]} '
            f'val_neutral={None if val_result["impact"] is None else val_result["impact"]["neutral_fraction"]} '
            f'val_base_MAE={None if val_result["impact"] is None else val_result["impact"]["base_MAE_norm"]} '
            f'val_final_MAE={None if val_result["impact"] is None else val_result["impact"]["final_MAE_norm"]} '
            f'val_mean_abs_error_change={None if val_result["impact"] is None else val_result["impact"]["mean_abs_error_change_norm"]} '
            f'residual_scale={_residual_scale_value(model)}',
            flush=True,
        )

        if not checkpoint_eligible:
            continue
        if val_result['mae'] < best - min_delta:
            best = val_result['mae']
            best_logical_epoch = logical_epoch
            best_train_epoch = train_epoch
            stale = 0
            state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            store = snapshot_store(model)
            best_val_impact = copy.deepcopy(val_result.get('impact'))
        else:
            stale += 1
            if stale >= int(c['patience']):
                print(
                    f'EARLY STOP train_epoch={train_epoch} logical_epoch={logical_epoch} '
                    f'best_train_epoch={best_train_epoch} best_logical_epoch={best_logical_epoch}',
                    flush=True,
                )
                break

    if state is None or store is None:
        raise RuntimeError(
            f'{label} did not produce an eligible checkpoint; '
            f'first_eligible_epoch={first_eligible_epoch}'
        )

    model.load_state_dict(state)
    restore_store(model, store)
    restored_status = retrieval_store_status(model)
    if args.variant != 'native' and not restored_status['ready']:
        raise RuntimeError(
            'RAST matched-backbone restored best checkpoint has empty retrieval memory: '
            + json.dumps(restored_status, sort_keys=True)
        )
    print(
        f'RAST BEST CHECKPOINT train_epoch={best_train_epoch} '
        f'logical_epoch={best_logical_epoch} val_MAE_norm={best:.6f} '
        f'retrieval_store={json.dumps(restored_status, sort_keys=True)}',
        flush=True,
    )

    mean, std = scale_stats(cache)
    raw, final_impact = predict(
        model, val_loader, device, mean, std, best_logical_epoch,
        collect_impact=(args.variant != 'native'),
    )
    run_dir = make_run_dir(cfg, label, args.stage, args.seed, backbone_model=selected_backbone if args.variant != 'native' else None)
    method_key = 'RAST' if args.variant == 'native' else label
    torch.save(
        {
            'model': state,
            'best_epoch': best_logical_epoch,
            'best_train_epoch': best_train_epoch,
            'best_val_mae_norm': best,
            'variant': args.variant,
            'method_implementation_revision': cfg.get('method_implementation_revisions', {}).get(method_key),
            'reference_backbone_model': getattr(model, '_reference_backbone_model', None),
            'reference_backbone_checkpoint': getattr(model, '_reference_backbone_checkpoint', None),
            'reference_backbone_checkpoint_sha256': getattr(model, '_reference_backbone_checkpoint_sha256', None),
            'reference_backbone_seed': getattr(model, '_reference_backbone_seed', None),
            'safe_residual': bool(getattr(model, '_safe_residual', False)),
            'residual_scale': _residual_scale_value(model),
            'residual_scale_init': float(getattr(model, '_residual_scale_init', 0.0)),
            'residual_formula': ('base + alpha * (rast_fused - base)' if getattr(model, '_safe_residual', False) else None),
            'best_epoch_val_impact': best_val_impact if args.variant != 'native' else None,
            'final_val_impact_vs_reference_backbone': final_impact if args.variant != 'native' else None,
        },
        run_dir / 'best.pt',
    )
    save_store(run_dir / 'retrieval_store_best.npz', store)

    summary = save_val_outputs(
        cfg, cache, run_dir, raw, method_key, args.seed, args.stage,
        {
            'best_epoch': best_logical_epoch,
            'best_train_epoch': best_train_epoch,
            'best_val_MAE_norm': best,
            'params': sum(p.numel() for p in params),
            'wall_seconds': time.time() - t0,
            'official_architecture': 'vendor/RAST/src/arch/rast_arch.py::RAST',
            'variant': args.variant,
            'retrieval_memory': 'training batches only',
            'checkpoint_selection': (
                'val_only'
                if args.variant == 'native'
                else 'val_only_after_retrieval_ready'
            ),
            'selection_min_delta': min_delta,
            'first_checkpoint_eligible_epoch': first_eligible_epoch,
            'retrieval_store_best': restored_status,
            'retrieval_first_warmup': bool(args.variant != 'native' and retrieval_first),
            'retrieval_prewarm_epoch': prewarm_epoch if args.variant != 'native' else None,
            'retrieval_prewarm_store': prewarm_status,
            'correction_diagnostics': 'final_prediction_minus_reference_backbone_on_valid_targets',
            'best_epoch_val_impact': best_val_impact if args.variant != 'native' else None,
            'final_val_impact_vs_reference_backbone': final_impact if args.variant != 'native' else None,
            'impact_diagnostic_contract': 'matched_reference_backbone_helpful_harmful_v1',
            'test_access': False,
            'reference_backbone_model': getattr(model, '_reference_backbone_model', None),
            'reference_backbone_seed': getattr(model, '_reference_backbone_seed', None),
            'reference_backbone_checkpoint': getattr(model, '_reference_backbone_checkpoint', None),
            'reference_backbone_checkpoint_sha256': getattr(model, '_reference_backbone_checkpoint_sha256', None),
            'safe_residual': bool(getattr(model, '_safe_residual', False)),
            'residual_scale': _residual_scale_value(model),
            'residual_scale_init': float(getattr(model, '_residual_scale_init', 0.0)),
            'residual_formula': ('base + alpha * (rast_fused - base)' if getattr(model, '_safe_residual', False) else None),
        },
    )
    if args.variant != 'native':
        with open(run_dir / 'rast_reference_backbone_impact.json', 'w') as f:
            json.dump(final_impact, f, indent=2, sort_keys=True)
            f.write('\n')
    print(
        'RAST COMPLETE',
        json.dumps(
            {
                'run_dir': str(run_dir),
                'val': summary['val'],
                'impact': final_impact if args.variant != 'native' else None,
            },
            default=str,
        ),
        flush=True,
    )


if __name__ == '__main__':
    main()
