"""Training / prediction loops for ReShift-STLA v0.2.

Efficiency choices (see README section "效率"):
  * failure-replay shifts and the counterfactual target are precomputed once per
    split and shared by every run (`replay_bank.py`): no teacher forward in a step;
  * splits are GPU-resident (`DeviceBatches`), no DataLoader workers;
  * candidate scorers are factorised (`FactorizedScorer`);
  * `training.train_limit` / `val_limit` / `val_every` and `--profile-steps` for quick iteration.
"""
from __future__ import annotations
import json, random, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from .data import SupportSplit, FullTrainRepository, DeviceBatches
from .stla import load_pretrained_stla, MaskAwareSTLA, FrozenTeacher
from .model import ReShiftSTLA
from .memory import build_memory_index
from .graph import graph_neighbours
from .seasonal import train_future_reference
from .teacher import prepare_pfull_bank
from .replay_bank import prepare_query_bank
from .losses import compute_losses, alignment_loss, parts_dict
from .evaluation import metrics, save_summary
from .checkpoint import persist_epoch, restore_training

MODEL_NAME = 'ReShift-STLA'
COVERAGE_BINS = (('unobserved', 0.0, 0.0), ('low', 1e-9, 0.5), ('high', 0.5 + 1e-9, 1.0 - 1e-9), ('complete', 1.0, 1.0))


def seed_all(seed, deterministic=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic); torch.backends.cudnn.benchmark = not deterministic


def phase(ep, cfg):
    tc = cfg['training']
    if tc.get('schedule') == 'frozen' or ep <= int(tc['warmup_epochs']): return 'warmup'
    if ep <= int(tc['partial_epochs']): return 'partial'
    return 'full'


def build_model(cfg, checkpoint, device):
    base, ck = load_pretrained_stla(Path(cfg['project']['root']), Path(checkpoint), device)
    teacher = FrozenTeacher(base)            # copied BEFORE any fine-tuning
    model = ReShiftSTLA(MaskAwareSTLA(base), teacher, int(ck['num_nodes']), int(cfg['experiment']['history']),
                        int(cfg['experiment']['horizon']), cfg).to(device)
    return model, ck


def setup_graph(model, cfg, cache, force=False):
    rc = cfg['retrieval']
    path = Path(cfg['project']['store_root']) / cfg['experiment']['dataset'] / 'train_only_graph.npz'
    model.set_graph(graph_neighbours(cache, path, int(rc['graph_neighbors']), int(rc.get('graph_sampled_rows', 1200)),
                                     force=force, lag_steps=rc.get('graph_lag_steps', [0, 1, 2, 3])))
    return path


def optimizer(model, cfg):
    tc = cfg['training']
    frozen = {id(p) for m in (model.teacher, model.memory_encoder) for p in m.parameters()}
    bb = [p for p in model.stla.base.parameters()]; bids = {id(p) for p in bb}
    new = [p for p in model.parameters() if id(p) not in bids and id(p) not in frozen]
    return torch.optim.AdamW([{'params': new, 'lr': float(tc['lr_new'])}, {'params': bb, 'lr': float(tc['lr_backbone'])}],
                             weight_decay=float(tc['weight_decay']))


def move(b, device): return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def select_rows(n, limit, seed):
    """All rows, or a fixed random subset (sorted) of `limit` rows for quick iteration."""
    if not limit or int(limit) >= n: return np.arange(n, dtype=np.int64)
    return np.sort(np.random.default_rng(int(seed)).choice(n, int(limit), replace=False)).astype(np.int64)


def make_batches(ds, rows, bs, shuffle, cfg, device, smoke=False, seed=0):
    if torch.device(device).type == 'cuda' and cfg['runtime'].get('gpu_dataset', True) and not smoke:
        try:
            return DeviceBatches(ds, rows, bs, shuffle, device, seed)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f'RESHIFT_DEVICE_SPLIT_FALLBACK split={ds.split}', flush=True)
    nw = 0 if smoke else int(cfg['runtime'].get('num_workers', 0))
    kw = dict(batch_size=bs, shuffle=shuffle, num_workers=nw,
              pin_memory=bool(cfg['runtime'].get('pin_memory', True)) and torch.cuda.is_available())
    if nw: kw.update(persistent_workers=True, prefetch_factor=int(cfg['runtime'].get('prefetch_factor', 2)))
    return DataLoader(Subset(ds, rows.tolist()), **kw)


def make_loader(ds, bs, shuffle, cfg, smoke=False):
    """Backward-compatible helper (all rows)."""
    return make_batches(ds, np.arange(len(ds)), bs, shuffle, cfg, 'cuda' if torch.cuda.is_available() else 'cpu', smoke)


def needs_pfull(model): return model.use_replay or model.use_cf


def bank_store(cfg): return Path(cfg['project']['store_root']) / cfg['experiment']['dataset']


def query_bank(model, cfg, split_name, ds, rows, repo, pfull, ck, device, smoke, with_cf):
    """Shared per-split bank: replay shifts for every row, plus CF observed forecasts for Train.

    The bank never depends on the ablation or on `train_limit`, so one build serves
    every run of a target (ablations only ignore the parts they do not use).
    `rows` is accepted for API symmetry; banks always cover the whole split.
    """
    if pfull is None: return None
    store = Path(cfg['project']['store_root']) / cfg['experiment']['dataset'] / 'banks'
    emb = int(cfg['replay'].get('train_embargo_steps', 0)) if split_name == 'train' else 0
    return prepare_query_bank(split_name + ('_smoke' if smoke else ''), ds, model.teacher, repo, pfull, ck, store, device,
                              int(cfg['replay']['top_r']), model.H + model.P, emb, split_name == 'train', None, smoke)


def attach_bank(b, model, bank, pfull, repo_row, device, training):
    if training and repo_row is not None and bank is None:
        b['repo_row'] = repo_row[b['index'].long().to(repo_row.device)]
    if bank is None: return
    if model.use_replay and bank.has('shift'):
        b['replay_shift'] = bank.fetch('shift', b['index'], device).float()
        b['replay_score'] = bank.fetch('score', b['index'], device).float()
    if training and repo_row is not None:
        b['repo_row'] = repo_row[b['index'].long().to(repo_row.device)]
    if training and model.use_cf and bank.has('pobs'):
        rr = repo_row[b['index'].long().to(repo_row.device)]
        b['cf_target'] = torch.nan_to_num(pfull.fetch(rr, device) - bank.fetch('pobs', b['index'], device).float())
        b['cf_target_valid'] = (rr >= 0)[:, None, None].expand_as(b['cf_target'])


def run_epoch(model, batches, index, repo, pfull, bank, repo_row, device, mean, std, cfg, opt=None, ep=1,
              scaler=None, max_steps=None, profile=False):
    training = opt is not None; model.train(training); cum = None; size = 0; skipped = 0
    amp = bool(cfg['runtime'].get('amp', False)) and device.type == 'cuda'
    align_on = training and model.uses_retrieval and float(cfg['training'].get('alignment_weight', 0)) > 0
    sync = (lambda: torch.cuda.synchronize(device)) if (profile and device.type == 'cuda') else (lambda: None)
    timing = {'data': 0., 'forward': 0., 'backward': 0.}; t_epoch = time.time(); t_last = time.time()
    with torch.enable_grad() if training else torch.no_grad():
        for step, bb in enumerate(batches, 1):
            b = move(bb, device); attach_bank(b, model, bank, pfull, repo_row, device, training)
            sync(); t0 = time.time(); timing['data'] += t0 - t_last
            if training: opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(b, index, repo, pfull)
                al = (alignment_loss(model, b, repo, device, float(cfg['training'].get('alignment_temperature', .1)), out)
                      if align_on else None)
                loss, parts = compute_losses(out, b, mean, std, cfg, al)
            sync(); t1 = time.time(); timing['forward'] += t1 - t0
            # No per-step loss check (it would force a host sync): non-finite gradients are caught
            # by the clip-norm check below, and the accumulated epoch losses are checked at the end.
            if training:
                trainable = [p for p in model.parameters() if p.requires_grad]
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward(); scaler.unscale_(opt)
                    norm = nn.utils.clip_grad_norm_(trainable, float(cfg['training']['grad_clip']))
                    if not bool(torch.isfinite(norm).item()):
                        skipped += 1; opt.zero_grad(set_to_none=True)
                        scaler.update(new_scale=max(1.0, float(scaler.get_scale()) / 2.0))
                        print(f'RESHIFT_GRAD_OVERFLOW epoch={ep} batch={step} skipped={skipped}', flush=True)
                        if skipped >= 20: raise RuntimeError('Repeated non-finite gradients')
                    else:
                        skipped = 0; old = scaler.get_scale(); scaler.step(opt); scaler.update()
                        if model.uses_retrieval and scaler.get_scale() >= old:
                            model.momentum_update(float(cfg['training'].get('memory_momentum', 0.995)))
                else:
                    loss.backward()
                    norm = nn.utils.clip_grad_norm_(trainable, float(cfg['training']['grad_clip']))
                    if not bool(torch.isfinite(norm).item()): raise RuntimeError(f'Non-finite gradients epoch={ep}')
                    opt.step()
                    if model.uses_retrieval: model.momentum_update(float(cfg['training'].get('memory_momentum', 0.995)))
            sync(); t_last = time.time(); timing['backward'] += t_last - t1
            n = int(b['x'].shape[0]); size += n
            cum = parts * n if cum is None else cum + parts * n
            if training and step % int(cfg['runtime']['print_every']) == 0:
                d = parts_dict(cum / size)
                print(f'RESHIFT_BATCH epoch={ep:03d} batch={step:05d}/{len(batches):05d} mae={d["mae"]:.5f} '
                      f'qps={size / (time.time() - t_epoch):.1f} '
                      f'gates=({d["gate_evolution"]:.3f},{d["gate_replay"]:.3f},{d["gate_cf"]:.3f})', flush=True)
            if max_steps and step >= max_steps: break
    res = parts_dict(cum / max(1, size))
    if not all(np.isfinite(v) for v in res.values()): raise RuntimeError(f'Non-finite epoch losses epoch={ep}: {res}')
    res['queries_per_second'] = size / max(time.time() - t_epoch, 1e-6)
    if profile: res.update({f'seconds_{k}': v for k, v in timing.items()}); res['steps'] = step
    return res


def predict(model, batches, index, repo, pfull, bank, device, mean, std, mode='full', amp=False):
    """Raw-unit predictions plus coverage-binned allocation diagnostics."""
    keys = ('prediction', 'base', 'true', 'valid', 'direct'); arr = {k: [] for k in keys}
    names = ('evolution', 'replay', 'cf')
    stat = {b: {'n': 0., 'gate': np.zeros(3), 'abs': np.zeros(3), 'gain': np.zeros(3)} for b, _, _ in COVERAGE_BINS}
    model.eval()
    with torch.no_grad():
        for bb in batches:
            b = move(bb, device); attach_bank(b, model, bank, pfull, None, device, False)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == 'cuda'):
                o = model(b, index, repo, pfull, mode=mode)
            pred = o['prediction'].float(); base = o['base_prediction'].float(); yn = b['y_norm'].float()
            arr['prediction'].append((pred * std + mean).cpu().numpy()); arr['base'].append((base * std + mean).cpu().numpy())
            arr['true'].append(b['y_raw'].cpu().numpy()); arr['valid'].append(b['future_valid'].cpu().numpy())
            arr['direct'].append(b['direct_mask'].cpu().numpy())
            # marginal raw-unit effect of each applied contribution: |P - Y| - |P - c_j - Y|  (< 0 helps)
            c = o['contributions'].float(); g = o['gates'].float(); v = b['future_valid'].bool()
            err = (pred - yn).abs()[..., None]; marg = (err - (pred[..., None] - c - yn[..., None]).abs()) * std
            cov = b['mask'].float().mean(1)[:, None, :].expand_as(pred)
            for name, lo, hi in COVERAGE_BINS:
                sel = v & (cov >= lo) & (cov <= hi)
                if not bool(sel.any()): continue
                s = stat[name]; s['n'] += float(sel.sum())
                s['gate'] += g[sel].sum(0).cpu().numpy(); s['abs'] += (c[sel].abs() * std).sum(0).cpu().numpy()
                s['gain'] += marg[sel].sum(0).cpu().numpy()
    out = {k: np.concatenate(v) for k, v in arr.items()}
    out['allocation'] = {b: ({'fraction_elements': s['n'],
                              **{f'gate_{n}': float(s['gate'][i] / s['n']) for i, n in enumerate(names)},
                              **{f'abs_correction_{n}': float(s['abs'][i] / s['n']) for i, n in enumerate(names)},
                              **{f'marginal_effect_{n}': float(s['gain'][i] / s['n']) for i, n in enumerate(names)}}
                             if s['n'] else None) for b, s in stat.items()}
    total = sum(s['n'] for s in stat.values()) or 1.
    for v in out['allocation'].values():
        if v: v['fraction_elements'] /= total
    return out


def refresh_index(model, repo, device, cfg, smoke):
    return build_memory_index(model.memory_encoder, repo, device, 16 if smoke else int(cfg['retrieval']['store_batch_size']))


def train(cfg, cache, checkpoint, run_dir, smoke=False, resume=None, profile_steps=0):
    tc = cfg['training']; seed = int(cfg['experiment']['seed'])
    seed_all(seed, bool(cfg['runtime'].get('deterministic', True)))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t_setup = time.time()
    model, ck = build_model(cfg, checkpoint, dev); graph_path = setup_graph(model, cfg, cache)
    mean = float(ck['mean']); std = float(ck['std'])
    tr = SupportSplit(cache, 'train', 24 if smoke else None); va = SupportSplit(cache, 'val', 24 if smoke else None)
    repo = FullTrainRepository(cache, 256 if smoke else None)
    model.set_future_reference(train_future_reference(repo, model.P, model.N))
    if not smoke and cfg['runtime'].get('preload_repository_gpu', True): repo.preload(dev)
    store = Path(cfg['project']['store_root']) / cfg['experiment']['dataset']
    pfull = (prepare_pfull_bank(model.teacher, repo, checkpoint, store, dev, smoke=smoke) if needs_pfull(model) else None)
    tr_rows = select_rows(len(tr), None if smoke else tc.get('train_limit'), seed)
    va_rows = select_rows(len(va), None if smoke else tc.get('val_limit'), seed + 1)
    tr_bank = query_bank(model, cfg, 'train', tr, tr_rows, repo, pfull, checkpoint, dev, smoke, with_cf=model.use_cf)
    va_bank = query_bank(model, cfg, 'val', va, va_rows, repo, pfull, checkpoint, dev, smoke, with_cf=False)
    # Train split row -> memory row (same complete window), via exact window_start.
    ws = tr.meta.window_start.to_numpy()[:len(tr)].astype(np.int64); pos = np.searchsorted(repo.starts, ws)
    pos = np.clip(pos, 0, len(repo.starts) - 1); repo_row = np.where(repo.starts[pos] == ws, pos, -1)
    repo_row = torch.from_numpy(repo_row).to(dev)
    bs = 3 if smoke else int(tc['batch_size'])
    tl = make_batches(tr, tr_rows, bs, True, cfg, dev, smoke, seed)
    vl = make_batches(va, va_rows, int(cfg['runtime'].get('eval_batch_size', bs)) if not smoke else bs, False, cfg, dev, smoke)
    opt = optimizer(model, cfg)
    scaler = torch.amp.GradScaler('cuda', enabled=bool(cfg['runtime'].get('amp', False)) and dev.type == 'cuda')
    print(f'RESHIFT_SETUP_PASS seconds={time.time() - t_setup:.1f} train_rows={len(tr_rows)} val_rows={len(va_rows)}', flush=True)
    index = None; best = float('inf'); best_ep = 0; stale = 0; prev = None; history = []; start = 1
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    if profile_steps:
        model.set_train_phase('partial', int(tc['partial_last_layers']))
        index = refresh_index(model, repo, dev, cfg, smoke).to(dev) if model.uses_retrieval else None
        if dev.type == 'cuda': torch.cuda.reset_peak_memory_stats(dev)
        r = run_epoch(model, tl, index, repo, pfull, tr_bank, repo_row, dev, mean, std, cfg, opt, 1, scaler,
                      max_steps=int(profile_steps), profile=True)
        r['peak_gpu_gib'] = torch.cuda.max_memory_allocated(dev) / 1024 ** 3 if dev.type == 'cuda' else 0.0
        r['estimated_epoch_minutes'] = len(tr_rows) / max(r['queries_per_second'], 1e-6) / 60
        print('RESHIFT_PROFILE ' + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                                               if not k.startswith(('evolution_', 'replay_'))}), flush=True)
        return None
    if resume:
        r = restore_training(resume, model, opt, scaler, cfg, checkpoint)
        start = int(r['epoch']) + 1; best = float(r['best_val_mae']); best_ep = int(r['best_epoch'])
        stale = int(r['stale']); prev = r['previous_phase']; history = list(r['history'])
    maxep = 2 if smoke else int(tc['max_epochs']); interval = int(cfg['retrieval']['store_refresh_epochs'])
    val_every = 1 if smoke else int(tc.get('val_every', 1))
    br = (model.use_evolution, model.use_replay, model.use_cf)
    print(f'RESHIFT_CONFIG ablation={tc["ablation"]} branches(evolution,replay,cf)={br} fusion={model.fusion_mode} '
          f'batch={bs} amp={scaler.is_enabled()} repository_gpu={repo._device} loader={type(tl).__name__}', flush=True)
    vidx = None
    for ep in range(start, maxep + 1):
        ph = phase(ep, cfg); model.set_train_phase(ph, int(tc['partial_last_layers']))
        if ph != prev: stale = 0; prev = ph; print(f'RESHIFT_PHASE epoch={ep} phase={ph}', flush=True)
        if model.uses_retrieval and (index is None or (ep - 1) % interval == 0):
            t0 = time.time(); index = refresh_index(model, repo, dev, cfg, smoke).to(dev)
            print(f'RESHIFT_MEMORY_REFRESH epoch={ep} seconds={time.time() - t0:.1f}', flush=True)
        if dev.type == 'cuda': torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.time(); a = run_epoch(model, tl, index, repo, pfull, tr_bank, repo_row, dev, mean, std, cfg, opt, ep, scaler)
        t_train = time.time() - t0
        if ep % val_every and ep != maxep: continue
        vidx = refresh_index(model, repo, dev, cfg, smoke).to(dev) if model.uses_retrieval else None
        v = run_epoch(model, vl, vidx, repo, pfull, va_bank, repo_row, dev, mean, std, cfg, None, ep, scaler)
        rec = {'epoch': ep, 'phase': ph, **{f'train_{k}': x for k, x in a.items()}, **{f'val_{k}': x for k, x in v.items()},
               'train_seconds': t_train, 'seconds': time.time() - t0,
               'peak_gpu_gib': torch.cuda.max_memory_allocated(dev) / 1024 ** 3 if dev.type == 'cuda' else 0.0}
        history.append(rec); print('RESHIFT_EPOCH ' + json.dumps(rec), flush=True)
        improved = v['mae'] < best - float(tc['min_delta'])
        if improved: best = v['mae']; best_ep = ep; stale = 0
        else: stale += val_every
        pd.DataFrame(history).to_csv(run_dir / 'history.csv', index=False)
        persist_epoch(run_dir, model, opt, scaler, cfg, checkpoint, ck, MODEL_NAME, ep, best, best_ep, history, stale, ph, improved)
        if improved and vidx is not None:
            vidx.save(run_dir / 'best_memory_index.npz', {'version': cfg['version'], 'rows': len(repo), 'graph': str(graph_path)})
        if stale >= int(tc['patience']) and (ph == 'full' or tc.get('schedule') == 'frozen'):
            print(f'RESHIFT_EARLY_STOP epoch={ep} best_epoch={best_ep}', flush=True); break
    best_path = run_dir / 'best.pt'
    if not best_path.is_file(): raise RuntimeError('No completed best checkpoint')
    model.load_state_dict(torch.load(best_path, map_location='cpu', weights_only=False)['state_dict'], strict=True); model.to(dev)
    index = None
    if model.uses_retrieval:
        index = refresh_index(model, repo, dev, cfg, smoke).to(dev)
        index.save(run_dir / 'best_memory_index.npz', {'version': cfg['version'], 'rows': len(repo), 'graph': str(graph_path)})
    pred = predict(model, vl, index, repo, pfull, va_bank, dev, mean, std, 'full', amp=scaler.is_enabled())
    vm = metrics(pred['prediction'], pred['true'], pred['valid'], pred['direct'])
    bm = metrics(pred['base'], pred['true'], pred['valid'], pred['direct'])
    save_summary(run_dir / 'summary.json', {'model': MODEL_NAME, 'version': cfg['version'], 'best_epoch': best_ep,
                 'ablation': tc['ablation'], 'validation': vm, 'base_validation': bm,
                 'validation_allocation': pred['allocation'], 'backbone_checkpoint': str(checkpoint), 'test_evaluated': False})
    print(f'RESHIFT_TRAIN_PASS run_dir={run_dir} best_epoch={best_ep} val_MAE={vm["MAE"]:.6f}', flush=True)
    return best_path
