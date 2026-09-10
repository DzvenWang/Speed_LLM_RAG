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

ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline')
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
    return [torch.from_numpy(a1).float().to(device), torch.from_numpy(a2).float().to(device)]


def import_model():
    vendor = ROOT / 'vendor' / 'GinAR'
    sys.path.insert(0, str(vendor))
    try:
        from model1.ginar_arch import GinAR
    except Exception as e:
        raise RuntimeError(f'GinAR official import failed from {vendor}: {e!r}') from e
    return GinAR


def run_epoch(model, loader, device, optimizer=None, clip=0.0) -> float:
    train = optimizer is not None
    model.train(train)
    total = 0.0
    denom = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for feat, y, valid, _idx in loader:
            feat = feat.float().to(device, non_blocking=True)
            y = y.float().to(device, non_blocking=True)
            valid = valid.bool().to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            pred = model(feat)
            if pred.shape != y.shape:
                raise RuntimeError(f'GinAR output contract failed: pred={tuple(pred.shape)} y={tuple(y.shape)}')
            loss = masked_l1(pred, y, valid)
            if train:
                loss.backward()
                if clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
                optimizer.step()
            n = int(valid.sum().item())
            total += float(loss.detach().item()) * max(n, 1)
            denom += max(n, 1)
    return total / max(denom, 1)


def predict(model, loader, device, mean: float, std: float) -> np.ndarray:
    out = []
    model.eval()
    with torch.no_grad():
        for feat, _y, _valid, _idx in loader:
            pred = model(feat.float().to(device, non_blocking=True))
            out.append(pred.detach().cpu().numpy())
    pred_norm = np.concatenate(out, axis=0).astype(np.float32)
    return pred_norm * np.float32(std) + np.float32(mean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='development', choices=['development', 'formal'])
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--smoke', action='store_true', help='Run one train and one val batch only; do not save a formal result.')
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
    ).to(device)
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'GINAR START stage={args.stage} seed={seed} device={device} params={nparams}', flush=True)
    print(f'GINAR DATA train={len(train_ds)} val={len(val_ds)} H={train_ds.x.shape[1]} N={train_ds.x.shape[2]}', flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(rcfg['learning_rate']))
    milestones = set(int(x) for x in rcfg.get('lr_milestones', []))
    gamma = float(rcfg.get('lr_gamma', 0.5))
    max_epochs = 1 if args.smoke else int(rcfg['max_epochs'])
    patience = int(rcfg.get('patience', 20))
    clip = float(rcfg.get('gradient_clip', 5.0))

    best = float('inf'); best_state = None; best_epoch = 0; stale = 0
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        if args.smoke:
            # Minimal real official-model forward/backward contract.
            feat, y, valid, _ = next(iter(train_loader))
            optimizer.zero_grad(set_to_none=True)
            pred = model(feat.float().to(device)); loss = masked_l1(pred, y.float().to(device), valid.bool().to(device))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), clip); optimizer.step()
            vfeat, vy, vv, _ = next(iter(val_loader))
            model.eval()
            with torch.no_grad():
                vp = model(vfeat.float().to(device)); vl = masked_l1(vp, vy.float().to(device), vv.bool().to(device))
            print(f'GINAR GPU SMOKE PASS train_loss={loss.item():.6f} val_loss={vl.item():.6f} output={tuple(vp.shape)}', flush=True)
            return

        et = time.time()
        tr = run_epoch(model, train_loader, device, optimizer=optimizer, clip=clip)
        va = run_epoch(model, val_loader, device, optimizer=None)
        if epoch in milestones:
            for pg in optimizer.param_groups:
                pg['lr'] *= gamma
                pg['weight_decay'] = float(pg.get('weight_decay', 0.0)) * gamma
        lr = optimizer.param_groups[0]['lr']
        print(f'EPOCH {epoch:03d} train_MAE_norm={tr:.6f} val_MAE_norm={va:.6f} lr={lr:.8g} sec={time.time()-et:.1f}', flush=True)
        if va < best - 1e-8:
            best = va; best_epoch = epoch; stale = 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
            if stale >= patience:
                print(f'EARLY STOP epoch={epoch} best_epoch={best_epoch}', flush=True)
                break

    if best_state is None:
        raise RuntimeError('GinAR did not produce a checkpoint')
    model.load_state_dict(best_state); model.to(device)
    mean, std = scale_stats(cache)
    pred_raw = predict(model, val_loader, device, mean, std)
    run_dir = make_run_dir(cfg, 'GinAR', args.stage, seed)
    torch.save({'model': best_state, 'best_epoch': best_epoch, 'best_val_mae_norm': best}, run_dir / 'best.pt')
    summary = save_val_outputs(cfg, cache, run_dir, pred_raw, 'GinAR', seed, args.stage, {
        'best_epoch': int(best_epoch),
        'best_val_MAE_norm': float(best),
        'params': int(nparams),
        'wall_seconds': float(time.time() - t0),
        'official_architecture': 'vendor/GinAR/model1/ginar_arch.py::GinAR',
        'graph_support': 'official_doubletransition',
        'checkpoint_selection': 'val_only',
        'test_access': False,
    })
    print('GINAR COMPLETE', json.dumps({'run_dir': str(run_dir), 'val': summary['val']}, default=str), flush=True)


if __name__ == '__main__':
    main()
