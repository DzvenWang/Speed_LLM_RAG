#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd())
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from baseline.backbones.training_mode import set_mode_from_cli, current_mode, method_reference_parent
from baseline.backbones.reference_backbone import ReferenceBackbone, build_reference_backbone
from baseline.compat.lrtc_tmcp import recover_batch
from baseline.lrtc_causal_history import resolve_lrtc_canonical_history_source
from baseline.runners.common import ImputationDataset, make_run_dir, require_gpu, save_val_outputs, set_seed, validate_stage


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _split_provenance(cache: Path, split: str, source) -> dict:
    meta = pd.read_csv(cache / f'{split}_manifest.csv')
    idx_hash = source.split_index_hashes(meta.window_start.astype(np.int64).to_numpy())
    return {
        **idx_hash,
        'query_count': int(len(meta)),
        'min_window_start': int(meta.window_start.min()),
        'max_window_start': int(meta.window_start.max()),
        'x_obs_file_sha256': _sha256_file(cache / f'{split}_x_obs_norm.npy'),
        'history_observed_file_sha256': _sha256_file(cache / f'{split}_history_observed.npy'),
        'canonical_target_path': str(source.target_path),
        'canonical_valid_path': str(source.valid_path),
        'support_days': int(source.historical_days),
        'historical_days': int(source.historical_days),
        'context_days': int(source.context_days),
        'day_steps': int(source.day_steps),
        'history_intervals': int(source.history),
        'train_start': int(source.train_start),
        'train_end_exclusive': int(source.train_end_exclusive),
        'support_policy': str(source.support_policy),
        'tensor_layout': 'nodes_x_support_days_x_query_slots',
        'strict_future_isolation': True,
        'heldout_support_forbidden': True,
    }


def _complete_batch(
    cfg: dict,
    backbone: ReferenceBackbone,
    source,
    meta: pd.DataFrame,
    batch,
    device: torch.device,
    *,
    smoke: bool = False,
    progress_prefix: str = 'LRTC-TMCP',
):
    idx, x_norm, _full_norm, obs, _art, tod, dow = batch
    x_norm = x_norm.float()
    obs = obs.bool()
    tod = tod.long().to(device)
    dow = dow.long().to(device)

    mean = float(backbone.mean)
    std = float(backbone.std)
    x_raw = x_norm * std + mean

    values = []
    masks = []
    starts = []
    for j, qi in enumerate(idx.tolist()):
        st = int(meta.iloc[int(qi)].window_start)
        v, m, s = source.build_tensor(
            st,
            x_raw[j].numpy(),
            obs[j].numpy(),
        )
        values.append(v)
        masks.append(m)
        starts.append(s)

    tensor = torch.from_numpy(np.stack(values, axis=0)).float().to(device)  # [B,N,D,H]
    mask = torch.from_numpy(np.stack(masks, axis=0)).bool().to(device)
    rcfg = cfg['runners']['LRTC-TMCP']
    recovered, info = recover_batch(
        tensor,
        mask,
        rho=float(rcfg['rho']),
        theta0=float(rcfg['theta0']),
        beta=float(rcfg['beta']),
        gamma=float(rcfg['gamma']),
        factor=float(rcfg['factor']),
        tol=float(rcfg['tol']),
        max_iter=int(rcfg['smoke_max_iter'] if smoke else rcfg['max_iter']),
        decomposition=str(rcfg.get('decomposition', 'gram_eigh')),
        active_convergence=bool(rcfg.get('active_convergence', True)),
        progress_every=int(rcfg.get('smoke_progress_every' if smoke else 'progress_every', 1 if smoke else 10)),
        progress_prefix=progress_prefix,
    )

    # Only the final/current query-day slice is passed to the selected reference Backbone.
    comp_raw = recovered[:, :, -1, :].transpose(1, 2)        # [B,H,N]
    current_obs = obs.to(device)
    current_raw = x_raw.to(device)
    # Benchmark contract: observed cells are immutable.
    comp_raw = torch.where(current_obs, current_raw, comp_raw)
    comp_norm = (comp_raw - mean) / std
    return comp_norm, tod, dow, info, starts


@torch.no_grad()
def forecast_split(
    cfg: dict,
    cache: Path,
    split: str,
    backbone: ReferenceBackbone,
    device: torch.device,
    *,
    smoke: bool = False,
    return_provenance: bool = False,
):
    ds = ImputationDataset(cache, split)
    rcfg = cfg['runners']['LRTC-TMCP']
    source = resolve_lrtc_canonical_history_source(
        cfg, cache, split, mean=float(backbone.mean), std=float(backbone.std)
    )
    provenance = _split_provenance(cache, split, source)
    print('LRTC-TMCP INPUT CONTRACT', provenance, flush=True)

    dl = DataLoader(
        ds,
        batch_size=int(rcfg['batch_size']),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    out = []
    infos = []
    first_starts = None
    limit_batches = 1 if smoke else None
    total_batches = len(dl)
    completed_queries = 0
    split_t0 = time.time()
    for bi, batch in enumerate(dl):
        batch_size = int(batch[0].shape[0])
        batch_label = f'LRTC-TMCP BATCH {bi + 1}/{total_batches}'
        print(
            f'{batch_label} START queries={batch_size} completed={completed_queries}/{len(ds)}',
            flush=True,
        )
        batch_t0 = time.time()
        comp, tod, dow, info, starts = _complete_batch(
            cfg, backbone, source, ds.meta, batch, device, smoke=smoke,
            progress_prefix=batch_label,
        )
        pred = backbone.predict_raw(comp, tod, dow)
        if not torch.isfinite(pred).all():
            raise RuntimeError('LRTC-TMCP -> reference Backbone produced non-finite prediction')
        out.append(pred.cpu().numpy())
        infos.append(info)
        completed_queries += batch_size
        peak_mem_gb = (
            float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 3)
            if device.type == 'cuda' else 0.0
        )
        print(
            f'{batch_label} COMPLETE queries={batch_size} completed={completed_queries}/{len(ds)} '
            f'wall_seconds={time.time() - batch_t0:.2f} tmcp_seconds={info.elapsed_seconds:.2f} '
            f'iterations_max={info.max_query_iterations} iterations_mean={info.mean_query_iterations:.2f} '
            f'converged_fraction={info.converged_fraction:.4f} decomposition={info.decomposition} '
            f'peak_cuda_mem_gb={peak_mem_gb:.3f} split_elapsed={time.time() - split_t0:.2f}s',
            flush=True,
        )
        if first_starts is None and starts:
            first_starts = starts[0]
        if limit_batches is not None and bi + 1 >= limit_batches:
            break
    raw = np.concatenate(out, 0).astype(np.float32)
    if first_starts is not None:
        provenance['first_query_context_starts'] = [int(x) for x in first_starts]
    return (raw, infos, provenance) if return_provenance else (raw, infos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['development', 'formal'], required=True)
    ap.add_argument('--seed', type=int, default=10)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--backbone', required=True)
    ap.add_argument('--backbone-checkpoint', default=None)
    ap.add_argument('--backbone-train-mode', choices=['incomplete','complete'], default=None)
    a = ap.parse_args()
    set_mode_from_cli(a.backbone_train_mode)

    cfg, cache = validate_stage(a.stage, 'LRTC-TMCP')
    allowed = [int(x) for x in cfg['methods']['LRTC-TMCP'].get('formal_seeds', [10])]
    if a.stage == 'formal' and a.seed not in allowed:
        raise SystemExit(f'LRTC-TMCP is deterministic; formal seed must be one of {allowed}')
    set_seed(a.seed)
    device = require_gpu()
    backbone = build_reference_backbone(cfg, device, model=a.backbone, seed=int(cfg.get('reference_backbone',{}).get('seed',10)), checkpoint=a.backbone_checkpoint)

    vendor = ROOT / 'vendor' / 'LRTC-TMCP'
    for rel in ('lrtc-tmcp-demo.ipynb', 'utils.py'):
        if not (vendor / rel).exists():
            raise FileNotFoundError(vendor / rel)

    if a.smoke:
        raw, infos, prov = forecast_split(
            cfg, cache, 'val', backbone, device, smoke=True, return_provenance=True
        )
        print('LRTC-TMCP GPU SMOKE PASS', {
            'prediction_shape': tuple(raw.shape),
            'iterations': [x.iterations for x in infos],
            'tmcp_seconds': [x.elapsed_seconds for x in infos],
            'decomposition': [x.decomposition for x in infos],
            'tensor_layout': prov['tensor_layout'],
            'historical_days': prov['historical_days'],
            'context_days': prov['context_days'],
            'strict_future_isolation': prov['strict_future_isolation'],
            'reference_backbone_train_mode': backbone.train_mode, 'reference_backbone_model': backbone.model_name,
            'reference_backbone_seed': backbone.reference_seed,
            'reference_backbone_checkpoint': backbone.checkpoint_reference,
            'reference_backbone_checkpoint_sha256': backbone.checkpoint_sha256,
        }, flush=True)
        return

    t0 = time.time()
    raw, infos, prov = forecast_split(
        cfg, cache, 'val', backbone, device, smoke=False, return_provenance=True
    )
    rd = make_run_dir(cfg, 'LRTC-TMCP', a.stage, a.seed, backbone_model=backbone.model_name)
    marker = {
        'algorithm': 'LRTC-TMCP',
        'learned_parameters': False,
        'deterministic': True,
        'official_source': 'vendor/LRTC-TMCP/lrtc-tmcp-demo.ipynb',
        'tensor_layout': 'nodes_x_support_days_x_query_slots',
        'support_days': int(prov['support_days']),
        'historical_days': int(prov['historical_days']),
        'support_policy': prov['support_policy'],
        'strict_future_isolation': True,
        'heldout_support_forbidden': True,
        'method_implementation_revision': cfg.get('method_implementation_revisions', {}).get('LRTC-TMCP'),
        'reference_backbone_train_mode': backbone.train_mode, 'reference_backbone_model': backbone.model_name,
        'reference_backbone_seed': backbone.reference_seed,
        'reference_backbone_checkpoint': backbone.checkpoint_reference,
        'reference_backbone_checkpoint_sha256': backbone.checkpoint_sha256,
    }
    torch.save(marker, rd / 'best.pt')
    (rd / 'lrtc_input_provenance.json').write_text(json.dumps(prov, indent=2, sort_keys=True) + '\n')
    sm = save_val_outputs(cfg, cache, rd, raw, 'LRTC-TMCP', a.seed, a.stage, {
        'params': 0,
        'wall_seconds': time.time() - t0,
        'deterministic': True,
        'official_equation_source': 'vendor/LRTC-TMCP/lrtc-tmcp-demo.ipynb::svt_TMCP,recover_data',
        'compatibility_implementation': 'baseline/compat/lrtc_tmcp.py::recover_batch',
        'input_contract': (
            f"{prov['support_days']} complete same-slot H12 support slices from frozen Train only + "
            'current incomplete H12; no held-out support; no query future; observed current cells protected'
        ),
        'tensor_layout': prov['tensor_layout'],
        'support_days': int(prov['support_days']),
        'historical_days': int(prov['historical_days']),
        'context_days': int(prov['context_days']),
        'support_policy': prov['support_policy'],
        'support_train_start': int(prov['train_start']),
        'support_train_end_exclusive': int(prov['train_end_exclusive']),
        'heldout_support_forbidden': True,
        'method_implementation_revision': cfg.get('method_implementation_revisions', {}).get('LRTC-TMCP'),
        'input_provenance_file': str(rd / 'lrtc_input_provenance.json'),
        'query_window_start_sha256': prov['query_window_start_sha256'],
        'causal_context_start_sha256': prov['causal_context_start_sha256'],
        'x_obs_file_sha256': prov['x_obs_file_sha256'],
        'history_observed_file_sha256': prov['history_observed_file_sha256'],
        'mean_iterations_per_batch': float(np.mean([x.iterations for x in infos])) if infos else None,
        'mean_query_iterations': float(np.mean([x.mean_query_iterations for x in infos])) if infos else None,
        'mean_tmcp_seconds_per_batch': float(np.mean([x.elapsed_seconds for x in infos])) if infos else None,
        'tmcp_decomposition': str(cfg['runners']['LRTC-TMCP'].get('decomposition', 'gram_eigh')),
        'active_convergence': bool(cfg['runners']['LRTC-TMCP'].get('active_convergence', True)),
        'reference_backbone_train_mode': backbone.train_mode, 'reference_backbone_model': backbone.model_name,
        'reference_backbone_seed': backbone.reference_seed,
        'reference_backbone_checkpoint': backbone.checkpoint_reference,
        'reference_backbone_checkpoint_sha256': backbone.checkpoint_sha256,
        'checkpoint_selection': 'none_deterministic_completion',
        'test_access': False,
    })
    print('LRTC-TMCP COMPLETE', json.dumps({'run_dir': str(rd), 'val': sm['val']}, default=str), flush=True)


if __name__ == '__main__':
    main()
