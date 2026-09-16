from __future__ import annotations
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .data import CachedSplit
from .adapters import build_adapter
from .metrics import per_query_metrics, backbone_metric_summary, METRICS_REVISION
from .predictions import save_prediction_npz
from .utils import count_parameters, ensure_dir, log, seed_all, write_json
from .identity import load_cache_identity, require_cache_version_compatibility


def _assert_prediction_shape(pred, target, context):
    if tuple(pred.shape) != tuple(target.shape):
        raise RuntimeError(
            f'{context} prediction/target shape mismatch: '
            f'pred={tuple(pred.shape)} target={tuple(target.shape)}. '
            'Adapters must return the shared [B, P, N] forecast layout.'
        )


def _masked_training_loss(pred_raw, y_raw, valid, kind='mae'):
    m = valid.bool() & torch.isfinite(pred_raw) & torch.isfinite(y_raw)
    if not torch.any(m):
        return (pred_raw * 0).sum()
    p = pred_raw[m]
    y = y_raw[m]
    kind = str(kind).lower()
    if kind in ('mae', 'l1'):
        return torch.abs(p - y).mean()
    if kind in ('huber', 'smooth_l1'):
        return F.huber_loss(p, y, reduction='mean', delta=1.0)
    raise KeyError(f'Unknown train_loss={kind}')


def make_optimizer(model, mc):
    name = str(mc.get('optimizer', 'adam')).lower()
    common = dict(lr=float(mc['learning_rate']), weight_decay=float(mc.get('weight_decay', 0.0)))
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), eps=float(mc.get('adam_eps', 1e-8)), **common)
    if name == 'adamw':
        return torch.optim.AdamW(model.parameters(), eps=float(mc.get('adam_eps', 1e-8)), **common)
    if name == 'adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr_decay=float(mc.get('lr_decay', 0.0)),
            initial_accumulator_value=float(mc.get('initial_accumulator_value', 0.0)),
            weight_decay=float(mc.get('weight_decay', 0.0)),
            lr=float(mc['learning_rate']),
        )
    raise ValueError(f'Unknown optimizer={name}')


def make_scheduler(optimizer, mc):
    name = str(mc.get('scheduler', 'none')).lower()
    if name in ('none', 'null', ''):
        return None
    if name == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(x) for x in mc.get('milestones', [])],
            gamma=float(mc.get('lr_gamma', 0.1)),
        )
    if name == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(mc['step_size']),
            gamma=float(mc.get('lr_gamma', 0.9)),
        )
    if name == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=float(mc.get('lr_gamma', 0.5))
        )
    if name in ('plateau', 'reduceonplateau', 'reduce_lr_on_plateau'):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=float(mc.get('lr_gamma', 0.1)),
            patience=int(mc.get('scheduler_patience', 20)),
            threshold=float(mc.get('scheduler_threshold', 1e-8)),
            min_lr=float(mc.get('min_lr', 0.0)),
        )
    raise ValueError(f'Unknown scheduler={name}')


@torch.no_grad()
def evaluate(adapter, loader, meta, device):
    adapter.model.eval()
    preds, ys, valids, directs, supports, times = [], [], [], [], [], []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    for batch in loader:
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        t = time.perf_counter()
        pn = adapter.eval_forward(batch)
        _assert_prediction_shape(pn, batch['y_raw'], f'{adapter.name}/eval')
        pr = adapter.raw(pn)
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t)
        preds.append(pr.detach().cpu().numpy())
        ys.append(batch['y_raw'].numpy())
        valids.append(batch['future_valid'].numpy())
        directs.append(batch['direct_mask'].numpy())
        supports.append(batch['support_mask'].numpy())
    pred = np.concatenate(preds)
    y = np.concatenate(ys)
    valid = np.concatenate(valids)
    direct = np.concatenate(directs)
    support = np.concatenate(supports)
    rows = per_query_metrics(pred, y, valid, direct, support)
    df = pd.concat([meta.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    # Canonical aggregate metrics are micro-aggregated over valid target points.
    # Per-query metrics stay in the CSV for diagnostics only.
    summary = backbone_metric_summary(pred, y, valid, direct, support)
    fam_values = np.asarray(meta['family'].astype(str))
    for fam in pd.unique(fam_values):
        ii = np.flatnonzero(fam_values == str(fam))
        fs = backbone_metric_summary(pred[ii], y[ii], valid[ii], direct[ii], support[ii])
        for c, v in fs.items():
            summary[f'{c}__{fam}'] = float(v)
    summary['inference_seconds_total'] = float(sum(times))
    summary['inference_ms_per_query'] = float(1000 * sum(times) / max(len(df), 1))
    summary['peak_gpu_memory_mb'] = (
        float(torch.cuda.max_memory_allocated(device) / 1024 ** 2)
        if torch.cuda.is_available() else 0.0
    )
    return pred, df, summary


def _candidate_batch_sizes(mc):
    vals = [int(mc['batch_size'])]
    vals.extend(int(x) for x in mc.get('fallback_batch_sizes', []))
    out = []
    for x in vals:
        if x > 0 and x not in out:
            out.append(x)
    return out


def _train_once(model_name, cfg, cache_dir, run_dir, vendor_root, benchmark_adj, benchmark_nodes,
                mean, std, seed, stage, batch_size):
    seed_all(seed)
    run_dir = ensure_dir(run_dir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mc = dict(cfg['models'][model_name])
    mc['batch_size'] = int(batch_size)
    runtime = cfg['runtime']
    run_identity = load_cache_identity(cache_dir)
    require_cache_version_compatibility(cfg, run_identity, stage)
    write_json(run_dir / 'RUN_MANIFEST.json', {**run_identity, 'code_project_version': str(cfg.get('version')), 'model': model_name, 'seed': int(seed), 'status': 'STARTED'})

    train = CachedSplit(cache_dir, 'train')
    val = CachedSplit(cache_dir, 'val')
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True,
        num_workers=int(runtime['num_workers']), pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )

    adapter = build_adapter(
        model_name, vendor_root, train.x.shape[2], device,
        mean, std, mc, benchmark_adj, benchmark_nodes
    )
    model = adapter.model
    optimizer = make_optimizer(model, mc)
    scheduler = make_scheduler(optimizer, mc)
    params = count_parameters(model)
    lock_path = Path(vendor_root) / 'SOURCE_LOCK.json'
    if not lock_path.exists():
        raise RuntimeError(f'Missing source lock: {lock_path}')
    vendor = json.loads(lock_path.read_text())['repositories'][model_name]

    adapter_revision = getattr(adapter, 'adapter_revision', 'base')
    log(
        f'stage={stage} model={model_name} seed={seed} device={device} '
        f'batch_size={batch_size} params={params:,} vendor_commit={vendor["commit"]} '
        f'adapter_revision={adapter_revision}'
    )
    log(
        f'official_recipe={mc.get("recipe_source")} optimizer={mc.get("optimizer")} '
        f'lr={mc.get("learning_rate")} eps={mc.get("adam_eps")} '
        f'wd={mc.get("weight_decay")} scheduler={mc.get("scheduler")} '
        f'train_loss={mc.get("train_loss")}'
    )

    best = float('inf')
    best_epoch = -1
    wait = 0
    history = []
    batches_seen = 0
    total_start = time.time()
    max_epochs = int(mc['max_epochs'])
    patience = int(mc['patience'])
    grad = float(mc.get('grad_clip', 0.0))
    print_every = int(runtime['print_every_batches'])

    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        e0 = time.time()
        for bi, b in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            pn = adapter.train_forward(b, batches_seen)
            _assert_prediction_shape(pn, b['y_raw'], f'{model_name}/train')
            pr = adapter.raw(pn)
            y = b['y_raw'].to(device).float()
            valid = b['future_valid'].to(device)
            loss = _masked_training_loss(pr, y, valid, mc.get('train_loss', 'mae'))
            loss.backward()
            if grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            batches_seen += 1
            if bi % print_every == 0:
                log(
                    f'epoch={epoch:03d} batch={bi:05d}/{len(train_loader):05d} '
                    f'train_loss={np.mean(losses[-print_every:]):.6f}'
                )
        scheduler_name = str(mc.get('scheduler', 'none')).lower()
        if scheduler is not None and scheduler_name not in ('plateau', 'reduceonplateau', 'reduce_lr_on_plateau'):
            interval = max(1, int(mc.get('scheduler_step_interval', 1)))
            if epoch % interval == 0:
                scheduler.step()

        _, _, vs = evaluate(adapter, val_loader, val.meta, device)
        v = float(vs['MAE'])
        if scheduler is not None and scheduler_name in ('plateau', 'reduceonplateau', 'reduce_lr_on_plateau'):
            scheduler.step(v)
        rec = {
            'epoch': epoch,
            'train_loss': float(np.mean(losses)),
            'val_MAE': v,
            'val_support_dependent_MAE': float(vs['support_dependent_MAE']),
            'lr': float(optimizer.param_groups[0]['lr']),
            'seconds': float(time.time() - e0),
        }
        history.append(rec)
        log(
            f'epoch={epoch:03d} train_loss={rec["train_loss"]:.6f} '
            f'val_MAE={v:.6f} val_support_MAE={rec["val_support_dependent_MAE"]:.6f} '
            f'lr={rec["lr"]:.3e} time={rec["seconds"]:.1f}s'
        )
        if v < best - 1e-8:
            best = v
            best_epoch = epoch
            wait = 0
            torch.save({
                'project_version': cfg.get('version', 'unknown'),
                'run_identity': run_identity,
                'stage': stage,
                'model': model_name,
                'state_dict': model.state_dict(),
                'model_config': mc,
                'num_nodes': train.x.shape[2],
                'mean': mean,
                'std': std,
                'seed': seed,
                'batch_size': batch_size,
                'vendor': vendor,
                'adapter_revision': adapter_revision,
                'adapter_metadata': getattr(adapter, 'adapter_metadata', {}),
                'best_val_MAE': best,
                'best_epoch': best_epoch,
            }, run_dir / 'best.pt')
        else:
            wait += 1
            if wait >= patience:
                log(f'early_stop epoch={epoch} patience={patience}')
                break

    pd.DataFrame(history).to_csv(run_dir / 'history.csv', index=False)
    ck = torch.load(run_dir / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(ck['state_dict'])
    vp, vdf, vs = evaluate(adapter, val_loader, val.meta, device)
    save_prediction_npz(
        run_dir / 'val_predictions.npz',
        val.meta.view_id.astype(str).tolist(),
        vp.astype(np.float32),
        manifest_path=run_dir / 'val_predictions.manifest.json',
        metadata={
            'producer': 'Backbone', 'model': model_name, 'seed': int(seed),
            'stage': stage, 'split': 'val', 'code_project_version': str(cfg.get('version')),
            'run_identity': run_identity,
        },
    )
    vdf.to_csv(run_dir / 'val_metrics.csv', index=False)
    summary = {
        'project_version': cfg.get('version', 'unknown'),
        'run_identity': run_identity,
        'stage': stage,
        'model': model_name,
        'seed': seed,
        'parameters': params,
        'batch_size': batch_size,
        'best_epoch': best_epoch,
        'best_val_MAE': best,
        'train_seconds_total': float(time.time() - total_start),
        'vendor': vendor,
        'adapter_revision': adapter_revision,
        'adapter_metadata': getattr(adapter, 'adapter_metadata', {}),
        'recipe_source': mc.get('recipe_source'),
        'training_recipe': {
            k: mc.get(k) for k in [
                'optimizer', 'learning_rate', 'adam_eps', 'weight_decay',
                'scheduler', 'milestones', 'step_size', 'lr_gamma', 'scheduler_step_interval',
                'scheduler_patience', 'scheduler_threshold', 'min_lr',
                'max_epochs', 'patience', 'grad_clip', 'train_loss'
            ] if k in mc
        },
        'validation': vs,
        'metrics_revision': METRICS_REVISION,
        'test_evaluated': False,
    }
    write_json(run_dir / 'summary.json', summary)
    write_json(run_dir / 'RUN_MANIFEST.json', {**run_identity, 'code_project_version': str(cfg.get('version')), 'model': model_name, 'seed': int(seed), 'status': 'FINISHED', 'best_epoch': int(best_epoch), 'best_val_MAE': float(best)})
    log(
        f'FINISHED stage={stage} {model_name} seed={seed}: '
        f'val_MAE={vs["MAE"]:.6f} val_support={vs["support_dependent_MAE"]:.6f}; '
        'TEST NOT EVALUATED'
    )
    return summary


def train_model(model_name, cfg, cache_dir, run_dir, vendor_root, benchmark_adj, benchmark_nodes,
                mean, std, seed=10, stage='screening'):
    mc = cfg['models'][model_name]
    candidates = _candidate_batch_sizes(mc)
    last_error = None
    for idx, batch_size in enumerate(candidates):
        try:
            if idx:
                log(f'Retrying {model_name} with memory fallback batch_size={batch_size}')
            return _train_once(
                model_name, cfg, cache_dir, run_dir, vendor_root, benchmark_adj, benchmark_nodes,
                mean, std, seed, stage, batch_size
            )
        except torch.cuda.OutOfMemoryError as e:
            last_error = e
            log(f'CUDA OOM for {model_name} at batch_size={batch_size}')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Remove incomplete artifacts before a clean retry.
            for name in ['best.pt', 'history.csv', 'val_predictions.npz', 'val_predictions.manifest.json', 'val_metrics.csv', 'summary.json']:
                p = Path(run_dir) / name
                if p.exists():
                    p.unlink()
    if last_error is not None:
        raise last_error
    raise RuntimeError(f'No usable batch size for {model_name}')
