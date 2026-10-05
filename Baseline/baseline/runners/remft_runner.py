#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

GADI_ROOT = next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd())
ROOT = GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.remft_runtime import bind_remft_bert_env, resolve_remft_bert_path
from baseline.remft_native_history import resolve_remft_native_source

from baseline.runners.common import (
    make_run_dir,
    masked_mse,
    require_gpu,
    save_val_outputs,
    scale_stats,
    set_seed,
    validate_stage,
)


class ReMFTDataset(Dataset):
    """ReMFT native H96 input over the shared benchmark query.

    The final benchmark H12 is loaded verbatim from ``*_x_obs_norm.npy``.  Only
    the additional 84 causal steps are materialized locally, under the same
    sensing view family/severity.  This restores the native ReMFT receptive field
    without granting complete held-out states.
    """

    def __init__(self, cfg: dict, cache: Path, split: str, *, mean: float, std: float):
        self.cache = Path(cache)
        self.split = str(split)
        self.tail = np.load(self.cache / f'{split}_x_obs_norm.npy', mmap_mode='r')
        self.y = np.load(self.cache / f'{split}_y_norm.npy', mmap_mode='r')
        self.valid = np.load(self.cache / f'{split}_future_valid.npy', mmap_mode='r')
        self.meta = pd.read_csv(self.cache / f'{split}_manifest.csv')
        self.source = resolve_remft_native_source(cfg, mean=mean, std=std)
        self.nodes = int(self.tail.shape[2])
        if self.nodes != self.source.nodes:
            raise RuntimeError(f'ReMFT node mismatch cache={self.nodes} canonical={self.source.nodes}')
        if int(self.tail.shape[1]) != int(cfg['experiment']['history']):
            raise RuntimeError('ReMFT benchmark tail is not H12')
        if len(self.meta) != len(self.tail):
            raise RuntimeError('ReMFT manifest/cache row mismatch')

        starts = self.meta.window_start.astype(int).to_numpy()
        eligible = np.flatnonzero(starts >= self.source.prefix_len).astype(np.int64)
        if self.split != 'train' and len(eligible) != len(starts):
            raise RuntimeError(
                f'ReMFT native H96 would drop {len(starts)-len(eligible)} {self.split} queries; '
                'evaluation/validation alignment must remain exact'
            )
        self.indices = eligible if self.split == 'train' else np.arange(len(starts), dtype=np.int64)
        self.dropped_insufficient_history = int(len(starts) - len(self.indices))
        self.input_contract = {
            'native_seq_len': int(self.source.native_history),
            'benchmark_tail_len': int(self.source.benchmark_history),
            'prefix_len': int(self.source.prefix_len),
            'prefix_policy': 'same_view_prefix_rematerialization_plus_exact_benchmark_tail',
            'train_rows_dropped_for_insufficient_prefix': self.dropped_insufficient_history,
            'heldout_hidden_values_exposed': False,
        }

    def __len__(self):
        return int(len(self.indices))

    def __getitem__(self, i):
        qi = int(self.indices[int(i)])
        row = self.meta.iloc[qi]
        x, _audit = self.source.build(
            window_start=int(row.window_start),
            view_id=str(row.view_id),
            tail_x_obs_norm=np.asarray(self.tail[qi], dtype=np.float32),
        )
        # Official ReMFT convention is [sensor, time].
        return (
            np.array(x.T, np.float32, copy=True),
            np.array(self.y[qi].T, np.float32, copy=True),
            np.array(self.valid[qi].T, np.uint8, copy=True),
        )


@contextmanager
def chdir(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def load_official_namespace() -> dict:
    """Execute only official ReMFT architecture cells.

    Stock dataset/training/test notebook cells are intentionally not executed.
    """
    nb_path = ROOT / 'vendor/ReMFT/run.ipynb'
    obj = json.loads(nb_path.read_text())
    ns = {'__name__': 'remft_official_arch'}
    executed = 0
    with chdir(nb_path.parent):
        for cell in obj.get('cells', []):
            if cell.get('cell_type') != 'code':
                continue
            code = ''.join(cell.get('source', []))
            bert_dir = bind_remft_bert_env(ROOT)
            bert_literal = str(bert_dir).replace('\\', '/')
            # Official ReMFT architecture uses a repo-relative local BERT path.
            # Redirect only that local checkpoint reference to the resolved shared cache.
            code = code.replace('./LLM/BERT', bert_literal).replace('LLM/BERT', bert_literal)
            if 'data_file = os.path.join' in code or '# load data' in code:
                break
            if not code.strip():
                continue
            exec(compile(code, f'{nb_path.name}:cell{executed}', 'exec'), ns, ns)
            executed += 1
    for name in ['MultiScaleSeasonMixing', 'MultiScaleTrendMixing', 'PHI', 'llmodel']:
        if name not in ns:
            raise RuntimeError(f'Official ReMFT notebook architecture missing {name}')
    return ns


def build_phi(ns: dict, cfg: dict) -> nn.Module:
    """Build the official PHI at its native 96-step working scale."""
    c = cfg['runners']['ReMFT']
    H = int(c.get('native_seq_len', 96))
    ds_window = int(c['down_sampling_window'])
    ds_layers = int(c['down_sampling_layers'])
    if H != 96:
        raise RuntimeError(f'ReMFT main-track correction requires native_seq_len=96, got {H}')
    scale = ds_window ** ds_layers
    if H % scale != 0:
        raise ValueError(
            f'ReMFT native history={H} must be divisible by '
            f'down_sampling_window**down_sampling_layers={scale}'
        )
    PHI = ns['PHI']
    phi = PHI(
        seq_len=H,
        k=ds_layers,
        c=ds_window,
        layers=int(c['phi_layers']),
    )
    # Official PHI constructs both multi-scale mixers with default seq_len=96.
    # At the restored native H96 scale they are already correct; do not rebuild
    # or reshape the architecture as the former H12 adapter did.
    if int(phi.mixing_multi_scale_season.seq_len) != H:
        raise RuntimeError('ReMFT official season mixer is not at native seq_len=96')
    if int(phi.mixing_multi_scale_trend.seq_len) != H:
        raise RuntimeError('ReMFT official trend mixer is not at native seq_len=96')
    return phi


class ReMFT96(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        c = cfg['runners']['ReMFT']
        H = int(c.get('native_seq_len', 96))
        P = int(cfg['experiment']['horizon'])
        ns = load_official_namespace()
        self.phi = build_phi(ns, cfg)
        repo = ROOT / 'vendor/ReMFT'
        with chdir(repo):
            self.llmodel = ns['llmodel'](
                patch_len=int(c['patch_len']),
                stride=int(c['stride']),
                pred_len=P,
                seq_len=H,
            )

    def forward(self, x):
        # Native ReMFT path: incomplete H96 history -> PHI -> frozen-BERT reprogramming.
        return self.llmodel(self.phi(x))


def _checkpoint_depth(bert_dir: Path) -> int:
    cfg = json.loads((bert_dir / 'config.json').read_text())
    depth = int(cfg.get('num_hidden_layers', 0))
    if depth <= 0:
        raise RuntimeError('ReMFT local BERT config has invalid num_hidden_layers')
    return depth


def normalize_bert_depth_to_checkpoint(model: nn.Module) -> dict:
    """Remove vendor-created random BERT layers beyond the pinned checkpoint depth.

    The pinned local checkpoint is authoritative for the pretrained frozen BERT.
    This does not change pretrained layers or the ReMFT mechanism; it prevents
    randomly initialized, frozen layers that have no checkpoint provenance from
    being used by the baseline.
    """
    bert_dir = resolve_remft_bert_path(ROOT)
    target_depth = _checkpoint_depth(bert_dir)
    llm = model.llmodel.llm_model
    encoder = getattr(llm, 'encoder', None)
    if encoder is None or not hasattr(encoder, 'layer'):
        raise RuntimeError('ReMFT BERT encoder.layer not found')
    runtime_before = len(encoder.layer)
    if runtime_before < target_depth:
        raise RuntimeError(
            f'ReMFT runtime BERT depth {runtime_before} is shallower than checkpoint depth {target_depth}'
        )
    trimmed = 0
    if runtime_before > target_depth:
        encoder.layer = nn.ModuleList(list(encoder.layer[:target_depth]))
        trimmed = runtime_before - target_depth
    runtime_cfg = getattr(llm, 'config', None)
    if runtime_cfg is not None and hasattr(runtime_cfg, 'num_hidden_layers'):
        runtime_cfg.num_hidden_layers = target_depth
    meta = {
        'runtime_before': runtime_before,
        'checkpoint_depth': target_depth,
        'trimmed_random_layers': trimmed,
    }
    print('REMFT BERT DEPTH NORMALIZED ' + ' '.join(f'{k}={v}' for k, v in meta.items()), flush=True)
    return meta


def llm_contract_metadata(model: nn.Module) -> dict:
    bert_dir = resolve_remft_bert_path(ROOT)
    local_cfg = json.loads((bert_dir / 'config.json').read_text())
    llm = model.llmodel.llm_model
    runtime_cfg = getattr(llm, 'config', None)
    params = list(llm.parameters())
    frozen = bool(params) and all(not p.requires_grad for p in params)
    if not frozen:
        trainable = [name for name, p in llm.named_parameters() if p.requires_grad]
        raise RuntimeError(
            'ReMFT frozen-LLM contract failed; BERT has trainable parameters: '
            + str(trainable[:20])
        )
    encoder = getattr(llm, 'encoder', None)
    encoder_layers = len(encoder.layer) if encoder is not None and hasattr(encoder, 'layer') else None
    runtime_layers = getattr(runtime_cfg, 'num_hidden_layers', None)
    meta = {
        'local_model_path': str(bert_dir),
        'local_num_hidden_layers': local_cfg.get('num_hidden_layers'),
        'local_hidden_size': local_cfg.get('hidden_size'),
        'local_num_attention_heads': local_cfg.get('num_attention_heads'),
        'runtime_num_hidden_layers': runtime_layers,
        'runtime_encoder_layer_count': encoder_layers,
        'runtime_hidden_size': getattr(runtime_cfg, 'hidden_size', None),
        'runtime_num_attention_heads': getattr(runtime_cfg, 'num_attention_heads', None),
        'llm_frozen': frozen,
        'runtime_matches_local_depth': (
            runtime_layers == local_cfg.get('num_hidden_layers')
            and encoder_layers == local_cfg.get('num_hidden_layers')
        ),
        'policy': 'checkpoint_consistent_pretrained_bert_depth',
    }
    print('REMFT LLM CONTRACT ' + json.dumps(meta, default=str), flush=True)
    if not meta['runtime_matches_local_depth']:
        raise RuntimeError('ReMFT runtime/checkpoint BERT depth mismatch after normalization')
    return meta


def build_model(cfg: dict, device: torch.device):
    repo = ROOT / 'vendor/ReMFT'
    bert = bind_remft_bert_env(ROOT)
    print(f'REMFT BERT PATH {bert}', flush=True)
    m = ReMFT96(cfg)
    m._depth_normalization = normalize_bert_depth_to_checkpoint(m)

    # The official notebook uses device_map='auto'. Keep BERT where the official
    # loader placed it; move only the trainable benchmark-side modules.
    llm_device = next(m.llmodel.llm_model.parameters()).device
    if llm_device != device:
        raise RuntimeError(
            f'ReMFT official BERT was placed on {llm_device}, expected {device}. '
            'Use the single-GPU PBS job and ensure the local BERT fits on that GPU.'
        )
    m.phi.to(device)
    for name in ['mapping_layer', 'patch_embedding', 'reprogramming_layer', 'output_projection']:
        getattr(m.llmodel, name).to(device)
    m._llm_contract = llm_contract_metadata(m)
    return m


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    sd = model.state_dict()
    return {k: v.detach().cpu().clone() for k, v in sd.items() if k in names}


def load_trainable_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f'ReMFT checkpoint has unexpected keys: {result.unexpected_keys[:10]}')


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {'0', 'false', 'no', 'off'}


def resolve_training_plan(c: dict, stage: str, train_rows: int) -> dict:
    """Resolve a reproducible compute plan while preserving effective batch size.

    Development remains exactly at the selected FP32 protocol. Formal can use a
    larger micro-batch + smaller accumulation, AMP, and a frozen optimizer-update
    budget derived from the completed Development selection.
    """
    dev_micro = int(c['batch_size'])
    dev_accum = max(1, int(c['gradient_accumulation_steps']))
    effective = int(c.get('effective_batch_size', dev_micro * dev_accum))
    if effective != dev_micro * dev_accum:
        raise RuntimeError(
            'ReMFT effective_batch_size must equal development batch_size * '
            'gradient_accumulation_steps so the completed Development protocol remains frozen'
        )

    if stage == 'development':
        micro = dev_micro
        eval_batch = int(c.get('eval_batch_size', dev_micro))
        accum = dev_accum
        use_amp = bool(c.get('development_use_amp', False))
        max_updates = None
    else:
        micro = int(os.getenv('REMFT_FORMAL_BATCH_SIZE', c.get('formal_batch_size', dev_micro)))
        eval_batch = int(os.getenv('REMFT_FORMAL_EVAL_BATCH_SIZE', c.get('formal_eval_batch_size', micro)))
        if effective % micro != 0:
            raise RuntimeError(
                f'ReMFT formal micro-batch {micro} must divide effective_batch_size={effective}'
            )
        accum = effective // micro
        use_amp = _bool_env('REMFT_FORMAL_AMP', bool(c.get('formal_use_amp', True)))
        cfg_budget = c.get('formal_max_optimizer_updates')
        env_budget = os.getenv('REMFT_FORMAL_MAX_UPDATES')
        max_updates = int(env_budget if env_budget is not None else cfg_budget) if (env_budget is not None or cfg_budget is not None) else None
        if max_updates is None or max_updates <= 0:
            raise RuntimeError('ReMFT Formal requires a positive fixed formal_max_optimizer_updates')
        budget_policy = str(c.get('formal_budget_policy', 'fixed_optimizer_updates'))
        if budget_policy != 'fixed_optimizer_updates':
            raise RuntimeError(f'Unsupported ReMFT formal_budget_policy={budget_policy!r}')

    workers = int(c.get('data_loader_workers', 2))
    plan = {
        'stage': stage,
        'train_rows': int(train_rows),
        'micro_batch_size': micro,
        'eval_batch_size': eval_batch,
        'effective_batch_size': effective,
        'gradient_accumulation_steps': accum,
        'use_amp': use_amp,
        'max_optimizer_updates': max_updates,
        'formal_budget_policy': str(c.get('formal_budget_policy', 'fixed_optimizer_updates')) if stage == 'formal' else None,
        'data_loader_workers': workers,
    }
    print('REMFT TRAINING PLAN ' + json.dumps(plan, sort_keys=True), flush=True)
    return plan


def _autocast(enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type='cuda', dtype=torch.float16)


def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)




def _optimizer_step(
    *,
    opt,
    scaler,
    trainable,
    gradient_clip: float,
    use_amp: bool,
) -> dict:
    """Apply one optimizer step with correct AMP overflow semantics.

    In AMP mode, GradScaler owns overflow handling: an inf/nan gradient causes
    scaler.step(optimizer) to skip optimizer.step(), then scaler.update() lowers
    the loss scale. Such a skipped step is not a fatal training error and must
    not count toward the frozen optimizer-update budget.

    In FP32 mode, non-finite gradients remain fatal because there is no dynamic
    loss scaling mechanism that can safely recover from them.
    """
    if use_amp:
        scaler.unscale_(opt)
        grad_norm = nn.utils.clip_grad_norm_(
            trainable,
            float(gradient_clip),
            error_if_nonfinite=False,
        )
        grad_finite = bool(torch.isfinite(torch.as_tensor(grad_norm)).item())
        scale_before = float(scaler.get_scale())
        scaler.step(opt)
        scaler.update()
        scale_after = float(scaler.get_scale())
        skipped = scale_after < scale_before

        # A non-finite gradient is expected to be absorbed by GradScaler. If the
        # scale did not back off, fail loudly rather than silently accepting an
        # optimizer state with undefined numerical behaviour.
        if (not grad_finite) and (not skipped):
            raise RuntimeError(
                'ReMFT non-finite AMP gradient was not skipped by GradScaler '
                f'(scale_before={scale_before}, scale_after={scale_after})'
            )
        return {
            'grad_norm': float(torch.as_tensor(grad_norm).detach().cpu()),
            'grad_finite': grad_finite,
            'skipped': bool(skipped),
            'scale_before': scale_before,
            'scale_after': scale_after,
        }

    grad_norm = nn.utils.clip_grad_norm_(
        trainable,
        float(gradient_clip),
        error_if_nonfinite=False,
    )
    grad_finite = bool(torch.isfinite(torch.as_tensor(grad_norm)).item())
    if not grad_finite:
        raise RuntimeError(f'ReMFT non-finite FP32 grad norm: {grad_norm}')
    opt.step()
    return {
        'grad_norm': float(torch.as_tensor(grad_norm).detach().cpu()),
        'grad_finite': True,
        'skipped': False,
        'scale_before': 1.0,
        'scale_after': 1.0,
    }


def _loader(ds: Dataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    kwargs = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )
    if workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)


def predict(model, dl, device, mean, std, use_amp: bool):
    model.eval()
    out = []
    with torch.no_grad():
        for x, _y, _v in dl:
            x = x.float().to(device, non_blocking=True)
            with _autocast(use_amp):
                p = model(x)  # [B,N,P]
            if not torch.isfinite(p).all():
                raise RuntimeError('ReMFT produced non-finite prediction')
            out.append(
                (p.transpose(1, 2).float().cpu().numpy() * np.float32(std) + np.float32(mean)).astype(np.float32)
            )
    return np.concatenate(out, 0)


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
    ap.add_argument('--stage', choices=['development', 'formal'], required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()

    cfg, cache = validate_stage(a.stage, 'ReMFT')
    set_seed(a.seed)
    device = require_gpu()
    c = cfg['runners']['ReMFT']

    mean, std = scale_stats(cache)
    tr = ReMFTDataset(cfg, cache, 'train', mean=mean, std=std)
    va = ReMFTDataset(cfg, cache, 'val', mean=mean, std=std)
    print('REMFT NATIVE INPUT CONTRACT ' + json.dumps({
        'train': tr.input_contract,
        'val': va.input_contract,
    }, sort_keys=True), flush=True)
    plan = resolve_training_plan(c, a.stage, len(tr))
    trl = _loader(tr, plan['micro_batch_size'], True, plan['data_loader_workers'])
    val = _loader(va, plan['eval_batch_size'], False, plan['data_loader_workers'])

    model = build_model(cfg, device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    nparams = sum(p.numel() for p in trainable)
    opt = torch.optim.Adam(
        trainable,
        lr=float(c['learning_rate']),
        weight_decay=float(c['weight_decay']),
        eps=1e-4,
    )
    accum = int(plan['gradient_accumulation_steps'])
    use_amp = bool(plan['use_amp'])
    scaler = _make_scaler(use_amp)

    if a.smoke:
        x, y, v = next(iter(trl))
        x = x.float().to(device, non_blocking=True)
        y = y.float().to(device, non_blocking=True)
        v = v.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with _autocast(use_amp):
            p = model(x)
            loss = masked_mse(p, y, v)
        if not torch.isfinite(loss):
            raise RuntimeError('ReMFT smoke produced non-finite loss')
        scaler.scale(loss).backward()
        step_meta = _optimizer_step(
            opt=opt,
            scaler=scaler,
            trainable=trainable,
            gradient_clip=float(c['gradient_clip']),
            use_amp=use_amp,
        )
        if step_meta['skipped']:
            raise RuntimeError('ReMFT smoke AMP step overflowed; retry smoke or lower smoke batch')
        exp = (int(x.shape[0]), tr.nodes, int(cfg['experiment']['horizon']))
        if tuple(p.shape) != exp or not torch.isfinite(p).all():
            raise RuntimeError(f'ReMFT smoke shape {tuple(p.shape)} != {exp}')
        print(
            'REMFT GPU SMOKE PASS',
            {
                'loss': float(loss.detach()),
                'shape': tuple(p.shape),
                'trainable_params': nparams,
                'plan': plan,
            },
            flush=True,
        )
        _save_lifecycle_smoke_checkpoint({'model': trainable_state_dict(model), 'best_epoch':1, 'checkpoint_format':'trainable_state_only_v1', 'lifecycle_smoke':True, 'method_implementation_revision': cfg.get('method_implementation_revisions', {}).get('ReMFT')})
        return

    best = float('inf')
    state = None
    be = 0
    stale = 0
    optimizer_updates = 0
    amp_skipped_steps = 0
    samples_seen = 0
    stop_for_budget = False
    t0 = time.time()

    for ep in range(1, int(c['max_epochs']) + 1):
        ep_t0 = time.time()
        model.train()
        opt.zero_grad(set_to_none=True)
        tt = tn = 0.0
        micro_since_step = 0

        for bi, (x, y, v) in enumerate(trl, 1):
            x = x.float().to(device, non_blocking=True)
            y = y.float().to(device, non_blocking=True)
            v = v.to(device, non_blocking=True)
            with _autocast(use_amp):
                p = model(x)
                loss = masked_mse(p, y, v)
            if not torch.isfinite(loss):
                raise RuntimeError(f'ReMFT non-finite loss epoch={ep} batch={bi}')

            scaler.scale(loss / accum).backward()
            tt += float(loss.detach().item())
            tn += 1
            samples_seen += int(x.shape[0])
            micro_since_step += 1

            should_step = micro_since_step >= accum or bi == len(trl)
            if should_step:
                step_meta = _optimizer_step(
                    opt=opt,
                    scaler=scaler,
                    trainable=trainable,
                    gradient_clip=float(c['gradient_clip']),
                    use_amp=use_amp,
                )
                opt.zero_grad(set_to_none=True)
                micro_since_step = 0

                if step_meta['skipped']:
                    amp_skipped_steps += 1
                    if amp_skipped_steps <= 10 or amp_skipped_steps % 50 == 0:
                        print(
                            f'REMFT AMP SKIP epoch={ep} batch={bi} '
                            f'grad_norm={step_meta["grad_norm"]} '
                            f'scale_before={step_meta["scale_before"]:.1f} '
                            f'scale_after={step_meta["scale_after"]:.1f} '
                            f'amp_skipped_steps={amp_skipped_steps}',
                            flush=True,
                        )
                else:
                    optimizer_updates += 1

                budget = plan['max_optimizer_updates']
                if budget is not None and optimizer_updates >= int(budget):
                    stop_for_budget = True
                    break

        model.eval()
        vt = vn = 0.0
        with torch.no_grad():
            for x, y, v in val:
                x = x.float().to(device, non_blocking=True)
                y = y.float().to(device, non_blocking=True)
                v = v.to(device, non_blocking=True)
                with _autocast(use_amp):
                    p = model(x)
                    loss = masked_mse(p, y, v)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'ReMFT non-finite validation loss epoch={ep}')
                vt += float(loss.detach().item())
                vn += 1

        score = vt / max(vn, 1)
        ep_sec = time.time() - ep_t0
        print(
            f'EPOCH {ep:03d} train_MSE_norm={tt/max(tn,1):.6f} '
            f'val_MSE_norm={score:.6f} optimizer_updates={optimizer_updates} '
            f'amp_skipped_steps={amp_skipped_steps} samples_seen={samples_seen} sec={ep_sec:.1f}',
            flush=True,
        )
        if score < best - 1e-8:
            best = score
            be = ep
            stale = 0
            state = trainable_state_dict(model)
        else:
            stale += 1
            if stale >= int(c['patience']):
                print(f'EARLY STOP epoch={ep} best_epoch={be}', flush=True)
                break

        if stop_for_budget:
            print(
                f'FORMAL UPDATE BUDGET REACHED optimizer_updates={optimizer_updates} '
                f'budget={plan["max_optimizer_updates"]} best_epoch={be}',
                flush=True,
            )
            break

    if state is None:
        raise RuntimeError('ReMFT produced no checkpoint')

    load_trainable_state_dict(model, state)
    raw = predict(model, val, device, mean, std, use_amp=use_amp)
    rd = make_run_dir(cfg, 'ReMFT', a.stage, a.seed)
    checkpoint = {
        'model': state,
        'checkpoint_format': 'trainable_state_only_v1',
        'best_epoch': be,
        'best_val_mse_norm': best,
        'history': int(c.get('native_seq_len', 96)),
        'benchmark_query_history': int(cfg['experiment']['history']),
        'horizon': int(cfg['experiment']['horizon']),
        'method_implementation_revision': cfg.get('method_implementation_revisions', {}).get('ReMFT'),
        'llm_contract': getattr(model, '_llm_contract', {}),
        'depth_normalization': getattr(model, '_depth_normalization', {}),
        'training_plan': plan,
        'optimizer_updates': optimizer_updates,
        'amp_skipped_steps': amp_skipped_steps,
        'samples_seen': samples_seen,
    }
    torch.save(checkpoint, rd / 'best.pt')

    sm = save_val_outputs(
        cfg,
        cache,
        rd,
        raw,
        'ReMFT',
        a.seed,
        a.stage,
        {
            'best_epoch': be,
            'best_val_MSE_norm': best,
            'trainable_params': nparams,
            'wall_seconds': time.time() - t0,
            'optimizer_updates': optimizer_updates,
            'amp_skipped_steps': amp_skipped_steps,
            'samples_seen': samples_seen,
            'training_plan': plan,
            'official_architecture': 'vendor/ReMFT/run.ipynb architecture cells through Model definition',
            'adaptation': (
                'official native H96 PHI + patch_len16/stride8 frozen-BERT reprogramming; '
                'extra causal H84 prefix is re-materialized under the same sensing view while '
                'the final benchmark H12 is reused exactly; no held-out hidden values are exposed; '
                'no stock data/train/test cells executed'
            ),
            'native_input_contract': va.input_contract,
            'table_group': cfg.get('methods', {}).get('ReMFT', {}).get('table_group', 'main'),
            'checkpoint_selection': 'Val masked MSE only',
            'llm_contract': getattr(model, '_llm_contract', {}),
            'depth_normalization': getattr(model, '_depth_normalization', {}),
            'test_access': False,
        },
    )
    print('REMFT COMPLETE', json.dumps({'run_dir': str(rd), 'val': sm['val']}, default=str))


if __name__ == '__main__':
    main()
