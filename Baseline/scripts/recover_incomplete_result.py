#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline') if Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline').exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline.evaluation import evaluate_prediction_file
from baseline.run_identity import build_run_manifest, write_run_manifest, load_cache_identity, require_same_identity
from baseline.runners.common import as_unicode_array
from baseline.shared_cache import cache_path
from baseline.utils import load_yaml, write_json


def newest_incomplete(base: Path) -> Path:
    cands = []
    if base.exists():
        for p in base.iterdir():
            if not p.is_dir():
                continue
            if (p / 'best.pt').exists() and (p / 'val_predictions.npz').exists() and not (p / 'summary.json').exists():
                cands.append(p)
    if not cands:
        raise SystemExit(f'No incomplete run with best.pt + val_predictions.npz under {base}')
    return max(cands, key=lambda p: p.stat().st_mtime)


def rewrite_prediction_npz(cache: Path, run_dir: Path) -> Path:
    pred_path = run_dir / 'val_predictions.npz'
    # The old artifact stores view_id as dtype=object. Access only pred with
    # allow_pickle=False, then rebuild IDs from the frozen manifest.
    with np.load(pred_path, allow_pickle=False) as z:
        pred = np.asarray(z['pred'], dtype=np.float32)
    meta = pd.read_csv(cache / 'val_manifest.csv')
    vids = as_unicode_array(meta.view_id.astype(str).tolist())
    tmp = run_dir / 'val_predictions.recovered.tmp.npz'
    np.savez_compressed(tmp, view_id=vids, pred=pred)
    os.replace(tmp, pred_path)
    return pred_path


def checkpoint_metadata(path: Path) -> dict:
    ck = torch.load(path, map_location='cpu', weights_only=False)
    out = {'best_epoch': int(ck.get('best_epoch', 0))}
    if 'best_val_mae_norm' in ck:
        out['best_val_MAE_norm'] = float(ck['best_val_mae_norm'])
    if 'best_val_mse_norm' in ck:
        out['best_val_MSE_norm'] = float(ck['best_val_mse_norm'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=['GinAR', 'SARAF'])
    ap.add_argument('--stage', default='development', choices=['development', 'formal'])
    ap.add_argument('--run-dir', default=None)
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(ROOT / 'configs' / 'baseline.yaml')
    exp = Path(cfg['project']['exp_root'])
    cache = cache_path(cfg, args.stage)
    base = exp / args.method / cfg['experiment']['subgraph'] / args.stage
    run_dir = Path(args.run_dir) if args.run_dir else newest_incomplete(base)
    if not run_dir.is_dir():
        raise SystemExit(f'run_dir not found: {run_dir}')

    expected_identity = load_cache_identity(cache, cfg['version'])
    existing_manifest = run_dir / 'RUN_MANIFEST.json'
    if not existing_manifest.exists():
        raise SystemExit('REFUSE: current-version recovery requires RUN_MANIFEST.json')
    require_same_identity(json.loads(existing_manifest.read_text()), expected_identity, label=str(existing_manifest))

    pred_path = rewrite_prediction_npz(cache, run_dir)
    metrics = evaluate_prediction_file(cache, 'val', pred_path)
    metrics.to_csv(run_dir / 'val_metrics.csv', index=False)
    all_row = metrics.loc[metrics.family.astype(str) == 'ALL'].iloc[0].to_dict()

    seed = int(cfg['experiment']['seed'] if args.seed is None else args.seed)
    ckmeta = checkpoint_metadata(run_dir / 'best.pt')
    manifest_extra = {
        'artifact_type': 'method_result',
        'runner_project_version': cfg['version'],
        'status': 'COMPLETE',
    }
    manifest = build_run_manifest(
        cache,
        cfg['version'],
        args.method,
        seed,
        cfg['methods'][args.method]['formal_track'],
        manifest_extra,
    )
    write_run_manifest(run_dir / 'RUN_MANIFEST.json', manifest)
    summary = {
        **manifest,
        'val': all_row,
        'prediction_file': str(pred_path),
        **ckmeta,
        'recovered_from_incomplete_run': True,
        'recovery_reason': 'current_version_incomplete_artifact_recovery',
        'selection_valid': True,
        'checkpoint_selection': 'val_only',
        'test_access': False,
    }
    write_json(run_dir / 'summary.json', summary)
    (base / 'LATEST_RUN.txt').write_text(str(run_dir) + '\n')
    print('RECOVERY PASS ' + json.dumps({'method': args.method, 'run_dir': str(run_dir), 'val': all_row}, default=str), flush=True)


if __name__ == '__main__':
    main()
