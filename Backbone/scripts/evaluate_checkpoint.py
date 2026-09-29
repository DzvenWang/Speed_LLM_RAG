#!/usr/bin/env python3
"""Evaluate an explicitly supplied Backbone checkpoint on a current evaluation cache.

This is intentionally lightweight.  It mirrors the original Backbone evaluation
behaviour: current evaluation tensors + current PatchSTG spatial layout + checkpoint
weights.  Historical metadata mismatches are logged, not used as blockers.
"""
from pathlib import Path
import argparse
import datetime
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(os.environ.get('SPEED_LLM_RAG_ROOT', ROOT.parent)).expanduser().resolve()
for p in (ROOT, REPO_ROOT / 'Base_SupportBench'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from backbone.data import CachedSplit
from backbone.adapters import build_adapter
from backbone.patchstg import compute_patchstg_layout
from backbone.predictions import save_prediction_npz
from backbone.runner import evaluate
from backbone.shared_cache import cache_path
from backbone.utils import load_yaml, log, sha256_file, write_json
from backbone.evaluation_reporting import scenario_rows, severity_row, write_result_json
from supportbench.dataset import SupportBench


def _resolve_checkpoint(value: str) -> Path:
    raw = Path(value).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    candidates = [
        (Path.cwd() / raw).resolve(),
        (REPO_ROOT / raw).resolve(),
        (ROOT / raw).resolve(),
    ]
    return next((p for p in candidates if p.exists()), candidates[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--model', default=None, choices=['STAEformer','HimNet','BigST','STDN','PatchSTG','STLAformer'], help='Optional model assertion; otherwise inferred from checkpoint')
    ap.add_argument('--severity', type=int, choices=[20, 40, 60, 90], required=True)
    ap.add_argument('--expected-seed', type=int, default=None)
    ap.add_argument('--config', default=str(ROOT / 'configs' / 'backbone.yaml'))
    ap.add_argument('--run-stamp', default=None)
    ap.add_argument('--result-json', default=None,
                    help='Optional compact machine-readable result for a parent multi-severity runner')
    ap.add_argument('--quiet-summary', action='store_true',
                    help='Suppress the standalone final table; parent runner will print a combined table')
    ap.add_argument('--smoke-test-batches', type=int, default=None,
                    help='Bound evaluation to this many real test batches; intended only for smoke tests')
    # Retained for CLI compatibility with v1.0.5.  They are intentionally ignored.
    ap.add_argument('--layout-snapshot', default=None)
    ap.add_argument('--allow-current-layout-mismatch', action='store_true')
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    checkpoint = _resolve_checkpoint(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')

    # Intentionally do not run Protocol access-gate, cache-identity, contract-revision,
    # formal/evaluation identity, or historical-layout compatibility checks here.
    # If the current evaluation cache is present and structurally usable, evaluate it.
    cache = cache_path(cfg, 'evaluation', args.severity)
    if not cache.exists():
        raise FileNotFoundError(f'Evaluation cache not found: {cache}')
    log(f'SIMPLE_EVALUATION severity={args.severity} cache={cache}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(ck, dict):
        raise RuntimeError(f'Checkpoint must contain a dict, got {type(ck)!r}')
    for key in ('state_dict', 'model_config', 'mean', 'std'):
        if key not in ck:
            raise RuntimeError(f'Checkpoint missing required key: {key}')
    if args.expected_seed is not None:
        got_seed = ck.get('seed')
        if got_seed is not None and int(got_seed) != int(args.expected_seed):
            log(f'WARNING checkpoint seed={got_seed} differs from expected_seed={args.expected_seed}; continuing')
    expected_target = str(cfg['experiment'].get('target','')).strip()
    checkpoint_target = str(ck.get('target') or ck.get('run_identity',{}).get('target_name') or '').strip()
    if checkpoint_target and expected_target and checkpoint_target != expected_target:
        raise RuntimeError(f'Checkpoint target={checkpoint_target} conflicts with evaluation target={expected_target}')
    log(f'TARGET_CONTRACT_PASS checkpoint_target={checkpoint_target or "legacy_unspecified"} evaluation_target={expected_target}')

    bench = SupportBench(cfg['experiment']['subgraph'], root=cfg['project']['benchmark_root'])
    test = CachedSplit(cache, 'test')

    # These are the only hard compatibility checks worth keeping: they prevent tensor
    # shape errors or silently evaluating the wrong network size.
    ck_nodes = int(ck.get('num_nodes', bench.N))
    if ck_nodes != int(bench.N):
        raise RuntimeError(f'Checkpoint num_nodes={ck_nodes} != current benchmark N={bench.N}')
    if int(test.x.shape[2]) != int(bench.N):
        raise RuntimeError(f'Evaluation cache N={test.x.shape[2]} != benchmark N={bench.N}')
    if int(test.x.shape[1]) != int(cfg['experiment']['history']):
        raise RuntimeError(f'Evaluation history={test.x.shape[1]} != configured history={cfg["experiment"]["history"]}')
    if int(test.y_raw.shape[1]) != int(cfg['experiment']['horizon']):
        raise RuntimeError(f'Evaluation horizon={test.y_raw.shape[1]} != configured horizon={cfg["experiment"]["horizon"]}')

    model_name = str(ck.get('model') or args.model or '')
    if not model_name:
        raise RuntimeError('Checkpoint does not declare model; pass --model explicitly')
    if args.model is not None and model_name != args.model:
        raise RuntimeError(f'Checkpoint model={model_name} conflicts with --model={args.model}')

    old_layout = None
    current_layout = None
    layout_match = True
    if model_name == 'PatchSTG':
        vendor_root = ROOT / 'vendor' / 'PatchSTG'
        _, _, _, current_layout = compute_patchstg_layout(
            vendor_root, bench.nodes, bench.adj, bench.N, ck['model_config']
        )
        if isinstance(ck.get('adapter_metadata'), dict):
            old_layout = ck['adapter_metadata'].get('spatial_layout_fingerprint')
        layout_match = (not old_layout) or str(old_layout) == str(current_layout)
        if not layout_match:
            log(
                'WARNING PatchSTG spatial layout fingerprint differs from historical checkpoint; '
                f'checkpoint={old_layout} current={current_layout}. Continuing with current SupportBench layout.'
            )

    adapter = build_adapter(
        model_name, ROOT / 'vendor', bench.N, device, float(ck['mean']), float(ck['std']),
        ck['model_config'], bench.adj, bench.nodes,
    )
    adapter.model.load_state_dict(ck['state_dict'], strict=True)
    log(
        f'CHECKPOINT_LOADED seed={ck.get("seed")} best_epoch={ck.get("best_epoch")} '
        f'layout_match={layout_match}'
    )

    batch_size = int(ck.get('batch_size', ck['model_config'].get('batch_size', 16)))
    eval_dataset = test
    eval_meta = test.meta
    if args.smoke_test_batches is not None:
        if int(args.smoke_test_batches) < 1:
            raise ValueError('--smoke-test-batches must be >= 1')
        max_queries = min(len(test), int(args.smoke_test_batches) * batch_size)
        eval_dataset = Subset(test, range(max_queries))
        eval_meta = test.meta.iloc[:max_queries].reset_index(drop=True)
        log(f'EVALUATION_SMOKE enabled batches={args.smoke_test_batches} queries={max_queries}')
    loader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    pred, df, summary = evaluate(adapter, loader, eval_meta, device)
    seed = int(ck.get('seed', args.expected_seed if args.expected_seed is not None else -1))
    families = [str(x) for x in cfg['experiment']['families']]
    query_counts = {str(k): int(v) for k, v in eval_meta['family'].astype(str).value_counts().to_dict().items()}
    sev_row = severity_row(summary, args.severity, seed)
    scen_rows = scenario_rows(summary, families, args.severity, seed, query_counts)

    stamp = args.run_stamp or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_sha = sha256_file(checkpoint)
    checkpoint_run_tag = checkpoint.parent.name or f'checkpoint_{checkpoint_sha[:12]}'
    out = (
        Path(cfg['project']['exp_root']) / model_name / cfg['experiment']['subgraph']
        / 'checkpoint_evaluation' / checkpoint_run_tag / f's{args.severity}'
        / f'{stamp}_seed{seed}'
    )
    out.mkdir(parents=True, exist_ok=True)

    metadata = {
        'producer': 'Backbone', 'model': model_name, 'seed': seed,
        'stage': 'checkpoint_evaluation', 'severity': args.severity,
        'smoke_test_batches': args.smoke_test_batches,
        'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_sha,
        'evaluation_policy': 'simple_current_cache_current_layout',
        'checkpoint_spatial_layout_fingerprint': old_layout,
        'current_spatial_layout_fingerprint': current_layout,
        'spatial_layout_match': bool(layout_match),
    }
    save_prediction_npz(
        out / 'test_predictions.npz', eval_meta.view_id.astype(str).tolist(), pred.astype(np.float32),
        manifest_path=out / 'test_predictions.manifest.json', metadata=metadata,
    )
    df.to_csv(out / 'test_metrics.csv', index=False)
    pd.DataFrame(scen_rows).to_csv(out / 'test_scenario_metrics.csv', index=False)
    write_json(out / 'test_summary.json', {
        'project_version': cfg.get('version'),
        'evaluation_mode': 'simple_explicit_checkpoint',
        'model': model_name, 'seed': seed, 'severity': args.severity,
        'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_sha,
        'checkpoint_best_epoch': ck.get('best_epoch'),
        'checkpoint_best_val_MAE': ck.get('best_val_MAE'),
        'evaluation_policy': 'current evaluation cache + current PatchSTG layout + checkpoint weights',
        'checkpoint_spatial_layout_fingerprint': old_layout,
        'current_spatial_layout_fingerprint': current_layout,
        'spatial_layout_match': bool(layout_match),
        'test': summary,
        'severity_summary': sev_row,
        'scenario_summary': scen_rows,
    })
    if args.result_json:
        write_result_json(
            args.result_json,
            severity=sev_row,
            scenarios=scen_rows,
            metadata={
                'output_dir': str(out),
                'checkpoint': str(checkpoint),
                'checkpoint_sha256': checkpoint_sha,
                'layout_match': bool(layout_match),
            },
        )
    log(
        f'CHECKPOINT EVALUATION COMPLETE s{args.severity} seed={seed} '
        f'MAE={summary["MAE"]:.6f} masked_MAE={summary["directly_masked_MAE"]:.6f} out={out}'
    )
    if not args.quiet_summary:
        from backbone.evaluation_reporting import severity_table_lines, scenario_table_lines
        print('\nFINAL SEVERITY SUMMARY', flush=True)
        for line in severity_table_lines([sev_row]): print(line, flush=True)
        print('\nFINAL SCENARIO SUMMARY', flush=True)
        for line in scenario_table_lines(scen_rows): print(line, flush=True)


if __name__ == '__main__':
    main()
