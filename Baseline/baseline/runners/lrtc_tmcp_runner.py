#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline')
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from baseline.backbones.patchstg_bridge import FrozenPatchSTG
from baseline.compat.lrtc_tmcp import recover_batch
from baseline.runners.common import ImputationDataset, make_run_dir, require_gpu, save_val_outputs, set_seed, validate_stage


def _complete_batch(cfg: dict, patch: FrozenPatchSTG, batch, device: torch.device, *, smoke: bool = False):
    _idx, x_norm, _full_norm, obs, _art, tod, dow = batch
    x_norm = x_norm.float().to(device)
    obs = obs.bool().to(device)
    tod = tod.long().to(device)
    dow = dow.long().to(device)

    mean = float(patch.mean)
    std = float(patch.std)
    # Work in raw speed space, as in the official LRTC traffic-completion code.
    x_raw = x_norm * std + mean
    # Missing entries are explicit through obs; zero is only a placeholder.
    sparse_raw = torch.where(obs, x_raw, torch.zeros_like(x_raw))

    # Strict main-table causality: each query is an independent
    # nodes x one-query-day x H12 tensor.  No other query or extra day is used.
    tensor = sparse_raw.transpose(1, 2).unsqueeze(2)          # [B,N,1,H]
    mask = obs.transpose(1, 2).unsqueeze(2)                  # [B,N,1,H]
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
    )
    comp_raw = recovered.squeeze(2).transpose(1, 2)          # [B,H,N]
    # Hard observed-value protection is part of the benchmark contract.
    comp_raw = torch.where(obs, x_raw, comp_raw)
    comp_norm = (comp_raw - mean) / std
    return comp_norm, tod, dow, info


@torch.no_grad()
def forecast_split(cfg: dict, cache: Path, split: str, patch: FrozenPatchSTG, device: torch.device, *, smoke: bool = False):
    ds = ImputationDataset(cache, split)
    rcfg = cfg['runners']['LRTC-TMCP']
    dl = DataLoader(ds, batch_size=int(rcfg['batch_size']), shuffle=False, num_workers=0, pin_memory=True)
    out = []
    infos = []
    limit_batches = 1 if smoke else None
    for bi, batch in enumerate(dl):
        comp, tod, dow, info = _complete_batch(cfg, patch, batch, device, smoke=smoke)
        pred = patch.predict_raw(comp, tod, dow)
        if not torch.isfinite(pred).all():
            raise RuntimeError('LRTC-TMCP -> PatchSTG produced non-finite prediction')
        out.append(pred.cpu().numpy())
        infos.append(info)
        if limit_batches is not None and bi + 1 >= limit_batches:
            break
    return np.concatenate(out, 0).astype(np.float32), infos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['development', 'formal'], required=True)
    ap.add_argument('--seed', type=int, default=10)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()

    cfg, cache = validate_stage(a.stage, 'LRTC-TMCP')
    allowed = [int(x) for x in cfg['methods']['LRTC-TMCP'].get('formal_seeds', [10])]
    if a.stage == 'formal' and a.seed not in allowed:
        raise SystemExit(f'LRTC-TMCP is deterministic; formal seed must be one of {allowed}')
    set_seed(a.seed)
    device = require_gpu()
    patch = FrozenPatchSTG(cfg, device)

    # Explicitly verify the pinned official repository is present.  We do not
    # modify it; the notebook remains the source-of-truth for the equations.
    vendor = ROOT / 'vendor' / 'LRTC-TMCP'
    for rel in ('lrtc-tmcp-demo.ipynb', 'utils.py'):
        if not (vendor / rel).exists():
            raise FileNotFoundError(vendor / rel)

    if a.smoke:
        raw, infos = forecast_split(cfg, cache, 'val', patch, device, smoke=True)
        print('LRTC-TMCP GPU SMOKE PASS', {
            'prediction_shape': tuple(raw.shape),
            'iterations': [x.iterations for x in infos],
            'strict_h12_only': True,
            'patchstg_checkpoint': str(patch.checkpoint),
        }, flush=True)
        return

    t0 = time.time()
    raw, infos = forecast_split(cfg, cache, 'val', patch, device, smoke=False)
    rd = make_run_dir(cfg, 'LRTC-TMCP', a.stage, a.seed)
    marker = {
        'algorithm': 'LRTC-TMCP',
        'learned_parameters': False,
        'deterministic': True,
        'official_source': 'vendor/LRTC-TMCP/lrtc-tmcp-demo.ipynb',
        'strict_h12_tensor_layout': 'nodes_x_single_query_day_x_history',
        'patchstg_reference_seed': patch.reference_seed,
        'patchstg_checkpoint': str(patch.checkpoint),
    }
    torch.save(marker, rd / 'best.pt')
    sm = save_val_outputs(cfg, cache, rd, raw, 'LRTC-TMCP', a.seed, a.stage, {
        'params': 0,
        'wall_seconds': time.time() - t0,
        'deterministic': True,
        'official_equation_source': 'vendor/LRTC-TMCP/lrtc-tmcp-demo.ipynb::svt_TMCP,recover_data',
        'compatibility_implementation': 'baseline/compat/lrtc_tmcp.py::recover_batch',
        'input_contract': 'current incomplete H12 only; no future, no extra days, no cross-query completion',
        'tensor_layout': cfg['runners']['LRTC-TMCP']['tensor_layout'],
        'mean_iterations_per_batch': float(np.mean([x.iterations for x in infos])) if infos else None,
        'patchstg_reference_seed': patch.reference_seed,
        'patchstg_checkpoint': str(patch.checkpoint),
        'checkpoint_selection': 'none_deterministic_completion',
        'test_access': False,
    })
    print('LRTC-TMCP COMPLETE', json.dumps({'run_dir': str(rd), 'val': sm['val']}, default=str), flush=True)


if __name__ == '__main__':
    main()
