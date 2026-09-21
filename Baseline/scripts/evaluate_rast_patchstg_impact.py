#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

GADI_ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline')
ROOT = GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.runners.common import RASTDataset, require_gpu, scale_stats, set_seed, validate_stage  # noqa: E402
from baseline.runners.rast_runner import (  # noqa: E402
    build_model,
    load_store,
    predict,
    restore_store,
    retrieval_store_status,
)


def read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, obj: dict) -> None:
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.write('\n')


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            'Compute RAFT/SARAF-compatible helpful/harmful diagnostics for an '
            'already-trained RAST-PatchSTG Formal run. No retraining is performed.'
        )
    )
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--seed', type=int, default=10)
    ap.add_argument('--update-summary', action='store_true')
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    ck_path = run_dir / 'best.pt'
    store_path = run_dir / 'retrieval_store_best.npz'
    for p in (ck_path, store_path):
        if not p.exists():
            raise FileNotFoundError(p)

    cfg, cache = validate_stage('formal', 'RAST')
    set_seed(args.seed)
    device = require_gpu()
    c = cfg['runners']['RAST']

    val_ds = RASTDataset(cache, 'val')
    val_loader = DataLoader(
        val_ds,
        batch_size=int(c['batch_size']),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    tmp = Path(cfg['project']['exp_root']) / '_tmp_rast' / f'RAST-PatchSTG-impact_{args.seed}_{int(time.time())}'
    tmp.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg, 'patchstg', args.seed, device, tmp)

    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    if str(ck.get('variant', 'patchstg')) != 'patchstg':
        raise RuntimeError(f'Expected patchstg checkpoint, got variant={ck.get("variant")}')
    state = ck.get('model')
    if not isinstance(state, dict):
        raise RuntimeError('best.pt does not contain a model state_dict')
    model.load_state_dict(state, strict=True)
    restore_store(model, load_store(store_path))
    status = retrieval_store_status(model)
    if not status['ready']:
        raise RuntimeError('Restored RAST retrieval store is not ready: ' + json.dumps(status, sort_keys=True))

    best_epoch = int(ck.get('best_epoch', 0))
    if best_epoch <= 0:
        raise RuntimeError(f'Invalid best_epoch={best_epoch} in {ck_path}')

    mean, std = scale_stats(cache)
    _raw_unused, impact = predict(
        model,
        val_loader,
        device,
        mean,
        std,
        best_epoch,
        collect_impact=True,
    )
    if impact is None:
        raise RuntimeError('RAST-PatchSTG impact collection returned no valid targets')

    result = {
        'method': 'RAST-PatchSTG',
        'diagnostic_contract': 'matched_patchstg_helpful_harmful_v1',
        'definitions_match': ['RAFT-PatchSTG', 'SARAF-PatchSTG'],
        'run_dir': str(run_dir),
        'checkpoint': str(ck_path),
        'retrieval_store': str(store_path),
        'best_epoch': best_epoch,
        'retrieval_store_status': status,
        'impact': impact,
    }
    out_path = run_dir / 'rast_patchstg_impact.json'
    write_json(out_path, result)

    if args.update_summary:
        summary_path = run_dir / 'summary.json'
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        backup = run_dir / f'summary.pre_rast_impact_{stamp}.json'
        shutil.copy2(summary_path, backup)
        summary = read_json(summary_path)
        summary['best_epoch_val_impact'] = impact
        summary['final_val_impact_vs_frozen_patchstg'] = impact
        summary['impact_diagnostic_contract'] = 'matched_patchstg_helpful_harmful_v1'
        summary['impact_diagnostic_source'] = 'posthoc_best_checkpoint_no_retraining'
        write_json(summary_path, summary)
        print(f'SUMMARY UPDATED backup={backup}', flush=True)

    print('RAST-PatchSTG IMPACT PASS ' + json.dumps(result, sort_keys=True, default=str), flush=True)
    print(f'IMPACT_JSON={out_path}', flush=True)


if __name__ == '__main__':
    main()
