#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

GADI_ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline')
ROOT = GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.backbones.patchstg_bridge import FrozenPatchSTG  # noqa: E402
from baseline.runners.common import (  # noqa: E402
    SARAFTensorDataset,
    build_time_marks,
    make_run_dir,
    masked_mse,
    require_gpu,
    save_val_outputs,
    scale_stats,
    set_seed,
    simple_namespace,
    validate_stage,
)


SUPPORTED = ('RAFT-PatchSTG', 'SARAF-PatchSTG')


from baseline.retrieval_patchstg_controls import (  # noqa: E402
    RAFTPatchSTGControl,
    SARAFPatchSTGControl,
)


def _correction_accumulate(stats: dict, final: torch.Tensor, base: torch.Tensor, y: torch.Tensor, valid: torch.Tensor) -> None:
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


def _empty_stats() -> dict:
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


def _finalize_stats(s: dict) -> dict:
    n = max(int(s['count']), 1)
    return {
        'count': int(s['count']),
        'correction_abs_mean_norm': float(s['corr_abs_sum'] / n),
        'correction_rms_norm': float(np.sqrt(s['corr_sq_sum'] / n)),
        'correction_max_abs_norm': float(s['corr_max_abs']),
        'helpful_fraction': float(s['helpful'] / n),
        'harmful_fraction': float(s['harmful'] / n),
        'neutral_fraction': float(s['neutral'] / n),
        # Negative means the matched retrieval control improved on PatchSTG.
        'mean_abs_error_change_norm': float(s['delta_error_sum'] / n),
        'base_MAE_norm': float(s['base_abs_error_sum'] / n),
        'final_MAE_norm': float(s['final_abs_error_sum'] / n),
    }


def _loader(ds, batch: int, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=True, drop_last=False)


def _run_epoch(model: nn.Module, dl: DataLoader, device: torch.device, mode: str, optimizer=None) -> tuple[float, dict]:
    train = optimizer is not None
    model.train(train)
    total = 0.0
    denom = 0
    stats = _empty_stats()
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for idx, x, y, xm, _ym in dl:
            idx = idx.long().to(device, non_blocking=True)
            x = x.float().to(device, non_blocking=True)
            y = y.float().to(device, non_blocking=True)
            xm = xm.float().to(device, non_blocking=True)
            ii = idx.detach().cpu().numpy()
            valid = torch.from_numpy(np.asarray(dl.dataset.valid[ii], np.uint8)).bool().to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            pred = model(x, xm, idx, mode=mode)
            base = model.last_base_prediction
            if base is None:
                raise RuntimeError('Matched PatchSTG control did not expose base prediction')
            if tuple(pred.shape) != tuple(y.shape):
                raise RuntimeError(f'Control output contract failed pred={tuple(pred.shape)} y={tuple(y.shape)}')
            loss = masked_mse(pred, y, valid)
            if not torch.isfinite(loss):
                raise FloatingPointError('Matched PatchSTG control produced non-finite loss')
            _correction_accumulate(stats, pred, base, y, valid)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 5.0
                )
                optimizer.step()
            n = int(valid.sum().item())
            total += float(loss.detach().item()) * max(n, 1)
            denom += max(n, 1)
    return total / max(denom, 1), _finalize_stats(stats)


def _predict(model: nn.Module, dl: DataLoader, device: torch.device, mean: float, std: float) -> tuple[np.ndarray, dict]:
    out = []
    stats = _empty_stats()
    model.eval()
    with torch.no_grad():
        for idx, x, y, xm, _ym in dl:
            idx = idx.long().to(device, non_blocking=True)
            x = x.float().to(device, non_blocking=True)
            y = y.float().to(device, non_blocking=True)
            xm = xm.float().to(device, non_blocking=True)
            ii = idx.detach().cpu().numpy()
            valid = torch.from_numpy(np.asarray(dl.dataset.valid[ii], np.uint8)).bool().to(device, non_blocking=True)
            pred = model(x, xm, idx, mode='valid')
            base = model.last_base_prediction
            if base is None:
                raise RuntimeError('Matched PatchSTG control did not expose base prediction during prediction')
            _correction_accumulate(stats, pred, base, y, valid)
            out.append(pred.detach().cpu().numpy())
    pn = np.concatenate(out, axis=0).astype(np.float32)
    return pn * np.float32(std) + np.float32(mean), _finalize_stats(stats)


def _build_raft(cfg, cache: Path, seed: int, device: torch.device, smoke: bool):
    from baseline.runners import raft_runner as native_runner

    rcfg = cfg['runners']['RAFT']
    txm, tym, time_source = build_time_marks(cfg, cache, 'train')
    vxm, vym, _ = build_time_marks(cfg, cache, 'val')
    lm = 256 if smoke else None
    lq = 64 if smoke else None
    memory = SARAFTensorDataset(cache, 'train', txm, tym, 'full', limit=lm)
    train = SARAFTensorDataset(cache, 'train', txm, tym, 'obs', limit=lq)
    val = SARAFTensorDataset(cache, 'val', vxm, vym, 'obs', limit=lq)
    H = int(cfg['experiment']['history'])
    P = int(cfg['experiment']['horizon'])
    N = int(train.x.shape[2])
    Model = native_runner.import_model()
    mc = simple_namespace(
        task_name='long_term_forecast', seq_len=H, pred_len=P, enc_in=N,
        n_period=int(rcfg['n_period']), topm=int(rcfg['topm']), gpu=0,
    )
    native = Model(mc).to(device)
    retrieval_info = native_runner.precompute(native, memory, train, val, cfg, device)
    frozen = FrozenPatchSTG(cfg, device)
    model = RAFTPatchSTGControl(native, frozen).to(device)
    return model, train, val, retrieval_info, time_source, rcfg


def _build_saraf(cfg, cache: Path, seed: int, device: torch.device, smoke: bool):
    from baseline.runners import saraf_runner as native_runner

    rcfg = cfg['runners']['SARAF']
    txm, tym, time_source = build_time_marks(cfg, cache, 'train')
    vxm, vym, _ = build_time_marks(cfg, cache, 'val')
    lm = 256 if smoke else None
    lq = 64 if smoke else None
    memory = SARAFTensorDataset(cache, 'train', txm, tym, query_signal='full', limit=lm)
    train = SARAFTensorDataset(cache, 'train', txm, tym, query_signal='obs', limit=lq)
    val = SARAFTensorDataset(cache, 'val', vxm, vym, query_signal='obs', limit=lq)
    H = int(cfg['experiment']['history'])
    P = int(cfg['experiment']['horizon'])
    N = int(train.x.shape[2])
    Model = native_runner.import_model()
    mc = simple_namespace(
        task_name='long_term_forecast', seq_len=H, pred_len=P,
        features=str(rcfg.get('features', 'M')), enc_in=N,
        n_period=int(rcfg.get('n_period', 1)), topm=int(rcfg.get('topm', 20)),
        freq=str(rcfg.get('freq', '5min')),
        time_aware_weight=float(rcfg.get('time_aware_weight', 0.3)),
        fusion_method=str(rcfg.get('fusion_method', 'weighted')),
        use_gpu=True, gpu=0,
    )
    native = Model(mc).to(device)
    retrieval_info = native_runner.precompute_retrieval(native, memory, train, val, cfg, device)
    frozen = FrozenPatchSTG(cfg, device)
    model = SARAFPatchSTGControl(native, frozen).to(device)
    return model, train, val, retrieval_info, time_source, rcfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=SUPPORTED)
    ap.add_argument('--stage', choices=['formal'], default='formal')
    ap.add_argument('--seed', type=int, default=10)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    native_key = 'RAFT' if args.method.startswith('RAFT-') else 'SARAF'
    cfg, cache = validate_stage(args.stage, native_key)
    set_seed(args.seed)
    device = require_gpu()

    if args.method == 'RAFT-PatchSTG':
        model, train_ds, val_ds, retrieval_info, time_source, rcfg = _build_raft(
            cfg, cache, args.seed, device, args.smoke
        )
    else:
        model, train_ds, val_ds, retrieval_info, time_source, rcfg = _build_saraf(
            cfg, cache, args.seed, device, args.smoke
        )

    batch = int(rcfg['batch_size'])
    train_dl = _loader(train_ds, batch, True)
    val_dl = _loader(val_ds, batch, False)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError(f'{args.method} has no trainable retrieval/fusion parameters')
    nparams = sum(p.numel() for p in params)
    optimizer = torch.optim.Adam(params, lr=float(rcfg['learning_rate']))

    print(
        f'{args.method} START stage={args.stage} seed={args.seed} device={device} '
        f'trainable_params={nparams} frozen_patchstg={model.frozen.checkpoint}',
        flush=True,
    )
    print(
        f'{args.method} CONTROL CONTRACT direct_forecaster=FrozenPatchSTG '
        f'retrieval={native_key} native_retriever_and_fusion query=incomplete train_memory=full_train',
        flush=True,
    )

    if args.smoke:
        tr, tr_stats = _run_epoch(model, train_dl, device, 'train', optimizer)
        va, va_stats = _run_epoch(model, val_dl, device, 'valid', None)
        print(
            f'{args.method} GPU SMOKE PASS train_MSE_norm={tr:.6f} val_MSE_norm={va:.6f} '
            f'train_stats={json.dumps(tr_stats, sort_keys=True)} '
            f'val_stats={json.dumps(va_stats, sort_keys=True)}',
            flush=True,
        )
        return

    max_epochs = int(rcfg['max_epochs'])
    patience = int(rcfg.get('patience', max_epochs))
    best = float('inf')
    best_epoch = 0
    best_state = None
    best_val_stats = None
    stale = 0
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        et = time.time()
        tr, tr_stats = _run_epoch(model, train_dl, device, 'train', optimizer)
        va, va_stats = _run_epoch(model, val_dl, device, 'valid', None)
        print(
            f'EPOCH {epoch:03d} train_MSE_norm={tr:.6f} val_MSE_norm={va:.6f} '
            f'val_helpful={va_stats["helpful_fraction"]:.4f} '
            f'val_harmful={va_stats["harmful_fraction"]:.4f} '
            f'val_corr_abs={va_stats["correction_abs_mean_norm"]:.6f} '
            f'sec={time.time()-et:.1f}',
            flush=True,
        )
        if va < best - 1e-10:
            best = va
            best_epoch = epoch
            stale = 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.native.state_dict().items()})
            best_val_stats = va_stats
        else:
            stale += 1
            if stale >= patience:
                print(f'EARLY STOP epoch={epoch} best_epoch={best_epoch}', flush=True)
                break

    if best_state is None:
        raise RuntimeError(f'{args.method} did not produce a checkpoint')
    model.native.load_state_dict(best_state, strict=True)
    model.to(device)
    model.frozen.adapter.model.eval()

    mean, std = scale_stats(cache)
    pred_raw, impact = _predict(model, val_dl, device, mean, std)
    run_dir = make_run_dir(cfg, args.method, args.stage, args.seed)
    checkpoint = {
        'method': args.method,
        'control_type': 'matched_frozen_patchstg_retrieval_control',
        'native_retrieval_method': native_key,
        'model': best_state,
        'best_epoch': int(best_epoch),
        'best_val_mse_norm': float(best),
        'frozen_patchstg_checkpoint': str(model.frozen.checkpoint),
        'frozen_patchstg_reference_seed': int(model.frozen.reference_seed),
        'retrieval_info': retrieval_info,
    }
    torch.save(checkpoint, run_dir / 'best.pt')

    summary = save_val_outputs(
        cfg, cache, run_dir, pred_raw, args.method, args.seed, args.stage,
        {
            'control_type': 'matched_frozen_patchstg_retrieval_control',
            'native_retrieval_method': native_key,
            'best_epoch': int(best_epoch),
            'best_val_MSE_norm': float(best),
            'trainable_params': int(nparams),
            'wall_seconds': float(time.time() - t0),
            'frozen_patchstg_checkpoint': str(model.frozen.checkpoint),
            'frozen_patchstg_reference_seed': int(model.frozen.reference_seed),
            'direct_forecast_path_replaced': 'native_linear_x -> FrozenPatchSTG',
            'native_retrieval_and_fusion_preserved': True,
            'retrieval': retrieval_info,
            'time_feature_source': time_source,
            'retrieval_memory_split': 'train',
            'retrieval_memory_signal': 'full_train_history',
            'query_signal': 'incomplete_history',
            'checkpoint_selection': 'val_only',
            'test_access': False,
            'best_epoch_val_impact': best_val_stats,
            'final_val_impact_vs_frozen_patchstg': impact,
        },
    )
    print(
        f'{args.method} COMPLETE ' + json.dumps(
            {'run_dir': str(run_dir), 'val': summary['val'], 'impact': impact},
            default=str,
        ),
        flush=True,
    )


if __name__ == '__main__':
    main()
