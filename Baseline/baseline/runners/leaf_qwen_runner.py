#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline')
import sys
sys.path.insert(0, str(ROOT))

from baseline.llm.leaf_qwen import (
    CHOICE_META,
    LocalQwenLeafSelector,
    build_choice_tensor,
    build_selector_prompt,
    multi_ranking_loss,
    selected_positive_and_negatives,
)
from baseline.runners.common import make_run_dir, require_gpu, save_val_outputs, scale_stats, set_seed, validate_stage
from baseline.runners.leaf_runner import import_leaf_models
from baseline.utils import write_json


class LeafQwenSplit:
    """Label-free selector/adaptation view of the frozen cache."""

    def __init__(self, cache: Path, split: str = 'val') -> None:
        self.cache = Path(cache); self.split = split
        self.x = np.load(self.cache / f'{split}_x_obs_norm.npy', mmap_mode='r')
        self.obs = np.load(self.cache / f'{split}_history_observed.npy', mmap_mode='r')
        self.tod_h = np.load(self.cache / f'{split}_tod_hist.npy', mmap_mode='r')
        self.dow_h = np.load(self.cache / f'{split}_dow_hist.npy', mmap_mode='r')
        self.tod_f = np.load(self.cache / f'{split}_tod_future.npy', mmap_mode='r')
        self.dow_f = np.load(self.cache / f'{split}_dow_future.npy', mmap_mode='r')
        self.meta = pd.read_csv(self.cache / f'{split}_manifest.csv')
        if len(self.meta) != len(self.x):
            raise RuntimeError('LEAF-Qwen cache/manifest row mismatch')

    def chronological_order(self) -> np.ndarray:
        if 'window_start' in self.meta.columns:
            return np.argsort(self.meta['window_start'].astype(np.int64).to_numpy(), kind='stable')
        return np.arange(len(self.x), dtype=np.int64)

    def batch(self, idx: int, device: torch.device) -> dict:
        x = np.asarray(self.x[idx], np.float32); h, n = x.shape
        tod = np.asarray(self.tod_h[idx], np.int64)[:, None].repeat(n, axis=1)
        dow = np.asarray(self.dow_h[idx], np.int64)[:, None].repeat(n, axis=1)
        return {
            'x': torch.from_numpy(x[..., None].copy()).unsqueeze(0).to(device),
            'time_of_day': torch.from_numpy(tod.copy()).unsqueeze(0).to(device),
            'day_of_week': torch.from_numpy(dow.copy()).unsqueeze(0).to(device),
        }


def _find_branch_run(cfg: dict, stage: str, seed: int) -> Path:
    base = Path(cfg['project']['exp_root']) / 'LEAF-branches' / cfg['experiment']['subgraph'] / stage
    candidates = sorted(base.glob(f'*_seed{int(seed)}'), key=lambda p: p.stat().st_mtime, reverse=True) if base.exists() else []
    for p in candidates:
        if (p / 'graph_best.pt').exists() and (p / 'hyper_best.pt').exists() and (p / 'summary.json').exists():
            try:
                sm = json.loads((p / 'summary.json').read_text())
            except Exception:
                continue
            if int(sm.get('seed', -1)) == int(seed) and str(sm.get('stage', '')) == str(stage):
                return p
    raise RuntimeError(f'No completed LEAF branch run for stage={stage} seed={seed} under {base}')


def _load_branch_models(cfg: dict, cache: Path, stage: str, seed: int, device: torch.device):
    branch_dir = _find_branch_run(cfg, stage, seed)
    summary_path = branch_dir / 'summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if int(summary.get('seed', seed)) != int(seed):
            raise RuntimeError(f'LEAF branch seed mismatch: requested {seed}, branch summary={summary.get("seed")}')
        if str(summary.get('stage', stage)) != str(stage):
            raise RuntimeError(f'LEAF branch stage mismatch: requested {stage}, branch summary={summary.get("stage")}')
    adj = np.load(cache / 'adjacency.npy')
    GraphBranch, HypergraphBranch = import_leaf_models(device, adj, cfg)
    graph = GraphBranch().to(device); hyper = HypergraphBranch().to(device)
    gck = torch.load(branch_dir / 'graph_best.pt', map_location='cpu')
    hck = torch.load(branch_dir / 'hyper_best.pt', map_location='cpu')
    graph.load_state_dict(gck['state_dict']); hyper.load_state_dict(hck['state_dict'])
    graph.to(device).train(); hyper.to(device).train()
    return branch_dir, adj, graph, hyper


def _optimizer(model: torch.nn.Module, lr: float, wd: float):
    params = list(model.get_finetune_params())
    if not params:
        raise RuntimeError(f'{model.__class__.__name__}.get_finetune_params() returned no parameters')
    return torch.optim.Adam(params, lr=float(lr), weight_decay=float(wd))


def _raw_from_model(model, batch, mean: float, std: float) -> torch.Tensor:
    out_norm = model(batch)
    if out_norm.ndim != 4 or out_norm.shape[0] != 1 or out_norm.shape[-1] != 1:
        raise RuntimeError(f'LEAF branch output contract failure: {tuple(out_norm.shape)}')
    return out_norm * float(std) + float(mean)


def _selected_raw(choices: torch.Tensor, selected_1based: np.ndarray) -> np.ndarray:
    # choices [12,P,N,1] -> [P,N]
    c = choices.detach().cpu().numpy()[..., 0]
    p, n = c.shape[1], c.shape[2]
    out = np.empty((p, n), np.float32)
    for node in range(n):
        out[:, node] = c[int(selected_1based[node]) - 1, :, node]
    return out


def _prompts_for_query(
    split: LeafQwenSplit,
    idx: int,
    choices: torch.Tensor,
    mean: float,
    std: float,
    degree: np.ndarray,
) -> list[str]:
    x_raw = np.asarray(split.x[idx], np.float32) * np.float32(std) + np.float32(mean)
    obs = np.asarray(split.obs[idx]).astype(bool)
    th = np.asarray(split.tod_h[idx], np.int64)
    dh = np.asarray(split.dow_h[idx], np.int64)
    tf = np.asarray(split.tod_f[idx], np.int64)
    df = np.asarray(split.dow_f[idx], np.int64)
    ch = choices.detach().cpu().numpy()[..., 0]  # 12,P,N
    prompts = []
    for node in range(ch.shape[2]):
        prompts.append(build_selector_prompt(
            node_idx=node,
            num_nodes=ch.shape[2],
            graph_degree=int(degree[node]),
            history_raw=x_raw[:, node],
            history_observed=obs[:, node],
            history_tod=th,
            history_dow=dh,
            future_tod=tf,
            future_dow=df,
            choices_raw=ch[:, :, node],
        ))
    return prompts


def _adapt_once(graph, hyper, batch, choices_raw, selected, mean, std, gopt, hopt, update_iters: int):
    positive, negatives = selected_positive_and_negatives(choices_raw, selected, mean, std)
    losses = []
    for _ in range(int(update_iters)):
        # Official LEAF only fine-tunes get_finetune_params(), i.e. LayerNorm parameters.
        gout = graph(batch)
        gopt.zero_grad(set_to_none=True)
        gl = multi_ranking_loss(gout, positive, negatives, margin=0.0)
        gl.backward(); gopt.step()

        hout = hyper(batch)
        hopt.zero_grad(set_to_none=True)
        hl = multi_ranking_loss(hout, positive, negatives, margin=0.0)
        hl.backward(); hopt.step()
        losses.append((float(gl.detach()), float(hl.detach())))
    return losses


def _checkpoint(path: Path, next_position: int, graph, hyper, gopt, hopt, selector_meta: dict, complete: bool = False):
    torch.save({
        'schema_version': 1,
        'next_position': int(next_position),
        'graph_state_dict': graph.state_dict(),
        'hyper_state_dict': hyper.state_dict(),
        'graph_optimizer': gopt.state_dict(),
        'hyper_optimizer': hopt.state_dict(),
        'selector_meta': selector_meta,
        'complete': bool(complete),
    }, path)


def _optimizer_to(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _load_checkpoint(path: Path, graph, hyper, gopt, hopt, device: torch.device) -> int:
    ck = torch.load(path, map_location='cpu')
    graph.load_state_dict(ck['graph_state_dict']); hyper.load_state_dict(ck['hyper_state_dict'])
    graph.to(device); hyper.to(device)
    gopt.load_state_dict(ck['graph_optimizer']); hopt.load_state_dict(ck['hyper_optimizer'])
    _optimizer_to(gopt, device); _optimizer_to(hopt, device)
    return int(ck['next_position'])


def _model_config_sha(snapshot: Path) -> str:
    h = hashlib.sha256()
    for name in ('config.json', 'tokenizer_config.json', 'generation_config.json'):
        p = snapshot / name
        if p.exists():
            h.update(name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['development', 'formal'], default='development')
    ap.add_argument('--seed', type=int, default=10)
    ap.add_argument('--max-queries', type=int, default=None, help='smoke/diagnostic only; production leaves this unset')
    ap.add_argument('--resume-dir', default=None)
    args = ap.parse_args()

    cfg, cache = validate_stage(args.stage, 'LEAF-Qwen1.5B')
    set_seed(args.seed); device = require_gpu()
    rc = cfg['runners']['LEAF']; qc = rc['qwen_selector']
    if int(qc['rounds']) != 2:
        raise RuntimeError('LEAF-Qwen1.5B is frozen to paper setting K=2')
    if int(qc['update_iters']) != 5:
        raise RuntimeError('LEAF-Qwen1.5B is frozen to official update_iters=5')

    branch_dir, adj, graph, hyper = _load_branch_models(cfg, cache, args.stage, args.seed, device)
    gopt = _optimizer(graph, float(qc['test_learning_rate']), float(qc['test_weight_decay']))
    hopt = _optimizer(hyper, float(qc['test_learning_rate']), float(qc['test_weight_decay']))
    mean, std = scale_stats(cache)
    split = LeafQwenSplit(cache, 'val')
    order = split.chronological_order()
    if args.max_queries is not None:
        order = order[:max(0, int(args.max_queries))]
    degree = (np.asarray(adj) > 0).sum(axis=1).astype(np.int64)

    if args.resume_dir:
        run_dir = Path(args.resume_dir).resolve()
        if not run_dir.is_dir(): raise RuntimeError(f'--resume-dir does not exist: {run_dir}')
    else:
        run_dir = make_run_dir(cfg, 'LEAF-Qwen1.5B', args.stage, args.seed)

    b, p, n = int(len(split.x)), int(cfg['experiment']['horizon']), int(adj.shape[0])
    pred_path = run_dir / 'val_qwen_raw.npy'
    r1_path = run_dir / 'round1_choices.npy'
    r2_path = run_dir / 'round2_choices.npy'
    inv_path = run_dir / 'round2_invalid.npy'
    if pred_path.exists():
        pred = np.lib.format.open_memmap(pred_path, mode='r+')
        r1 = np.lib.format.open_memmap(r1_path, mode='r+')
        r2 = np.lib.format.open_memmap(r2_path, mode='r+')
        invalid2 = np.lib.format.open_memmap(inv_path, mode='r+')
    else:
        pred = np.lib.format.open_memmap(pred_path, mode='w+', dtype=np.float32, shape=(b, p, n)); pred[:] = np.nan
        r1 = np.lib.format.open_memmap(r1_path, mode='w+', dtype=np.uint8, shape=(b, n)); r1[:] = 0
        r2 = np.lib.format.open_memmap(r2_path, mode='w+', dtype=np.uint8, shape=(b, n)); r2[:] = 0
        invalid2 = np.lib.format.open_memmap(inv_path, mode='w+', dtype=np.uint8, shape=(b, n)); invalid2[:] = 0

    selector = LocalQwenLeafSelector(
        model_cache_root=qc['model_cache_root'],
        snapshot=qc.get('snapshot'),
        device=device,
        batch_size=int(qc['selector_batch_size']),
        max_input_tokens=int(qc['max_input_tokens']),
        max_new_tokens=int(qc['max_new_tokens']),
        fallback_choice=int(qc['fallback_choice']),
    )
    print('LEAF-QWEN SELECTOR', json.dumps(selector.metadata(), indent=2), flush=True)
    print(f'LEAF-QWEN branch_run={branch_dir} queries={len(order)} total_val_rows={b} K=2 update_iters=5', flush=True)

    progress = run_dir / 'progress.pt'
    start_pos = 0
    if args.resume_dir and progress.exists():
        start_pos = _load_checkpoint(progress, graph, hyper, gopt, hopt, device)
        print(f'LEAF-QWEN RESUME next_position={start_pos}/{len(order)}', flush=True)

    t0 = time.time(); save_every = int(qc['checkpoint_every_queries']); loss_acc = []
    for pos in range(start_pos, len(order)):
        idx = int(order[pos])
        batch = split.batch(idx, device)
        selected_final = None; invalid_final = None; choices_final = None
        for round_idx in range(2):
            with torch.no_grad():
                graph_raw = _raw_from_model(graph, batch, mean, std)
                hyper_raw = _raw_from_model(hyper, batch, mean, std)
                choices = build_choice_tensor(graph_raw, hyper_raw)
            prompts = _prompts_for_query(split, idx, choices, mean, std, degree)
            selected, invalid = selector.select_prompts(prompts)
            if round_idx == 0:
                r1[idx] = selected
            else:
                r2[idx] = selected; invalid2[idx] = invalid
                selected_final = selected; invalid_final = invalid; choices_final = choices
            losses = _adapt_once(
                graph, hyper, batch, choices, selected, mean, std, gopt, hopt,
                update_iters=int(qc['update_iters']),
            )
            loss_acc.extend(losses)
        pred[idx] = _selected_raw(choices_final, selected_final)

        done = pos + 1
        if done % max(1, int(qc['log_every_queries'])) == 0 or done == len(order):
            recent = loss_acc[-max(1, int(qc['log_every_queries'])) * 10:]
            gmean = float(np.mean([z[0] for z in recent])) if recent else float('nan')
            hmean = float(np.mean([z[1] for z in recent])) if recent else float('nan')
            print(
                f'LEAF-QWEN progress={done}/{len(order)} cache_idx={idx} '
                f'rank_loss_graph={gmean:.6f} rank_loss_hyper={hmean:.6f} '
                f'invalid_total={selector.stats.invalid}/{selector.stats.total}', flush=True
            )
        if done % max(1, save_every) == 0:
            pred.flush(); r1.flush(); r2.flush(); invalid2.flush()
            _checkpoint(progress, done, graph, hyper, gopt, hopt, selector.metadata(), complete=False)

    pred.flush(); r1.flush(); r2.flush(); invalid2.flush()
    _checkpoint(progress, len(order), graph, hyper, gopt, hopt, selector.metadata(), complete=True)

    if len(order) != b:
        write_json(run_dir / 'smoke_summary.json', {
            'status': 'PARTIAL_MAX_QUERIES', 'processed_queries': len(order), 'total_val_rows': b,
            'selector': selector.metadata(), 'branch_run': str(branch_dir), 'test_access': False,
        })
        print(f'LEAF-QWEN PARTIAL PASS processed={len(order)}; canonical evaluation intentionally skipped', flush=True)
        return
    if np.isnan(np.asarray(pred)).any():
        raise RuntimeError('LEAF-Qwen final prediction contains NaN/unprocessed rows')

    counts = np.bincount(np.asarray(r2, dtype=np.int64).reshape(-1), minlength=13)[1:13]
    summary = save_val_outputs(
        cfg, cache, run_dir, np.asarray(pred, np.float32), 'LEAF-Qwen1.5B', args.seed, args.stage,
        {
            'variant': 'LEAF adapted to Qwen2.5-1.5B-Instruct',
            'adaptation_scope': 'original LEAF dual branches + official six transforms per branch + frozen LLM selection + Huber ranking adaptation',
            'original_selector': 'LLaMA-3-70B',
            'adapted_selector': 'Qwen2.5-1.5B-Instruct',
            'qwen_snapshot': str(selector.snapshot),
            'qwen_config_sha256': _model_config_sha(selector.snapshot),
            'branch_run': str(branch_dir),
            'choice_order_1based': [f'{bname}:{aug}' for bname, aug in CHOICE_META],
            'rounds_K': 2,
            'update_iters_M': int(qc['update_iters']),
            'ranking_margin': 0.0,
            'selector_stats': selector.stats.as_dict(),
            'round2_choice_counts': {str(i + 1): int(v) for i, v in enumerate(counts)},
            'round2_invalid_fraction': float(np.asarray(invalid2).mean()),
            'online_order': 'stable ascending val_manifest.window_start',
            'selector_inputs': 'history + history_observed + TOD/DOW + graph degree + 12 LEAF candidates; no y/ground_truth/errors/oracle',
            'test_access': False,
            'wall_seconds': time.time() - t0,
        },
    )
    print('LEAF-QWEN COMPLETE', json.dumps({'run_dir': str(run_dir), 'val': summary['val'], 'selector': selector.stats.as_dict()}, default=str), flush=True)


if __name__ == '__main__':
    main()
