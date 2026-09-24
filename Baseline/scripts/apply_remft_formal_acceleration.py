#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time

import yaml


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Patch only the ReMFT compute-plan keys in configs/baseline.yaml.'
    )
    ap.add_argument(
        '--config',
        default='configs/baseline.yaml',
        help='Path to Baseline config (default: configs/baseline.yaml)',
    )
    ap.add_argument('--formal-batch-size', type=int, default=8)
    ap.add_argument('--formal-eval-batch-size', type=int, default=8)
    ap.add_argument('--development-best-epoch', type=int, default=18)
    ap.add_argument('--development-train-rows', type=int, default=12000)
    ap.add_argument('--effective-batch-size', type=int, default=24)
    args = ap.parse_args()

    p = Path(args.config).resolve()
    if not p.exists():
        raise SystemExit(f'Config not found: {p}')
    obj = yaml.safe_load(p.read_text())
    c = obj.setdefault('runners', {}).setdefault('ReMFT', {})

    # Completed Development protocol remains frozen.
    dev_micro = int(c.get('batch_size', 1))
    dev_accum = int(c.get('gradient_accumulation_steps', 24))
    if dev_micro * dev_accum != args.effective_batch_size:
        raise SystemExit(
            f'Existing ReMFT Development effective batch is {dev_micro * dev_accum}, '
            f'not requested {args.effective_batch_size}. Refusing to silently change Development.'
        )
    if args.effective_batch_size % args.formal_batch_size != 0:
        raise SystemExit('formal-batch-size must divide effective-batch-size exactly')

    updates_per_dev_epoch = (
        args.development_train_rows + args.effective_batch_size - 1
    ) // args.effective_batch_size
    formal_budget = updates_per_dev_epoch * args.development_best_epoch

    c.update({
        'effective_batch_size': args.effective_batch_size,
        'development_use_amp': False,
        'formal_batch_size': args.formal_batch_size,
        'formal_eval_batch_size': args.formal_eval_batch_size,
        'formal_use_amp': True,
        'formal_reference_development_best_epoch': args.development_best_epoch,
        'formal_reference_development_train_rows': args.development_train_rows,
        'formal_max_optimizer_updates': formal_budget,
        'data_loader_workers': 2,
    })

    obj.setdefault('method_implementation_revisions', {})['ReMFT'] = (
        'remft_phi12_checkpointdepth_amp_batch8_overflowsafe_updatebudget_r4'
    )

    backup = p.with_suffix(p.suffix + f'.pre_remft_accel_{time.strftime("%Y%m%d_%H%M%S")}.bak')
    shutil.copy2(p, backup)
    p.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True))

    print('PATCHED', p)
    print('BACKUP ', backup)
    print('ReMFT Formal plan:', {
        'development_micro_batch': dev_micro,
        'development_accumulation': dev_accum,
        'effective_batch_size': args.effective_batch_size,
        'formal_micro_batch': args.formal_batch_size,
        'formal_accumulation': args.effective_batch_size // args.formal_batch_size,
        'formal_eval_batch': args.formal_eval_batch_size,
        'formal_amp': True,
        'development_best_epoch': args.development_best_epoch,
        'development_train_rows': args.development_train_rows,
        'formal_max_optimizer_updates': formal_budget,
    })


if __name__ == '__main__':
    main()
