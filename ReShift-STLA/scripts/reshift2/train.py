#!/usr/bin/env python3
"""Train one ReShift-STLA v0.2 configuration, then evaluate all severities (+ audit for `full`)."""
from pathlib import Path
import argparse, datetime, os, subprocess, sys
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from reshift2.config import load_config, apply_ablation, cache_path, resolve_stla_checkpoint, ABLATIONS
from reshift2.trainer import train


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--target', choices=['speed', 'flow']); p.add_argument('--seed', type=int, default=10)
    p.add_argument('--dataset'); p.add_argument('--backbone-checkpoint'); p.add_argument('--config')
    p.add_argument('--ablation', choices=ABLATIONS, default='full'); p.add_argument('--smoke', action='store_true')
    p.add_argument('--run-stamp'); p.add_argument('--batch-size', type=int); p.add_argument('--num-workers', type=int)
    p.add_argument('--resume', help='last.pt of the SAME experiment'); p.add_argument('--skip-evaluate', action='store_true')
    p.add_argument('--profile-steps', type=int, default=0, help='time N training steps, print RESHIFT_PROFILE and exit')
    p.add_argument('--audit', action='store_true', help='run in-checkpoint interventions after training (default for formal full runs)')
    a = p.parse_args()
    if a.target: os.environ['TRAFFIDENT_TARGET'] = a.target
    cfg = load_config(a.config); cfg['experiment']['seed'] = a.seed
    if a.dataset: cfg['experiment']['dataset'] = a.dataset
    if a.batch_size: cfg['training']['batch_size'] = a.batch_size
    if a.num_workers is not None: cfg['runtime']['num_workers'] = a.num_workers
    cfg = apply_ablation(cfg, a.ablation)
    cache = cache_path(cfg, 'formal'); ck = resolve_stla_checkpoint(cfg, a.backbone_checkpoint)
    tag = 'smoke' if a.smoke else ('profile' if a.profile_steps else Path(a.config or 'reshift2').stem.replace('reshift2', 'formal') or 'formal'); stamp = a.run_stamp or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run = (Path(a.resume).resolve().parent if a.resume else
           Path(cfg['project']['exp_root']) / cfg['experiment']['dataset'] / tag / f'seed{a.seed}_{a.ablation}_{stamp}')
    best = train(cfg, cache, ck, run, a.smoke, resume=a.resume, profile_steps=a.profile_steps)
    if best is None: return
    print(f'RESHIFT_BEST_CHECKPOINT={best}', flush=True)
    if a.skip_evaluate: return
    cmd = [sys.executable, '-u', str(ROOT / 'scripts/reshift2/evaluate.py'), '--checkpoint', str(best),
           '--backbone-checkpoint', str(ck), '--severities', *[str(s) for s in cfg['experiment']['severities']]]
    if a.smoke: cmd += ['--smoke']
    if a.audit or (a.ablation == 'full' and tag == 'formal'): cmd += ['--audit']
    subprocess.check_call(cmd, cwd=ROOT); print('RESHIFT_AUTO_EVALUATE_PASS', best, flush=True)


if __name__ == '__main__': main()
