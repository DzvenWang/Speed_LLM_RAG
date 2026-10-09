#!/usr/bin/env python3
"""Submit FARE Gadi PBS jobs from a login node.

By default, each target has ONE GPU job: train followed immediately by
checkpoint-based evaluation within the *same* PBS allocation. No held Evaluate
job is submitted. ``evaluate`` remains an independent manual option.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys

PROJECT_ROOT = Path(os.environ.get('SPEED_LLM_RAG_ROOT', Path(__file__).resolve().parents[2])).resolve()
PBS_DIR = PROJECT_ROOT / 'FARE' / 'pbs'
DATASETS = tuple(f'TraffiDent_Net_{i:02d}' for i in (1, 2, 3, 4, 6))
PHASES = ('retrieval', 'recovery', 'injection', 'partial', 'full', 'candidate', 'controller', 'best')


def to_qsub_env(values: dict) -> str:
    """qsub -v values cannot contain literal commas or line breaks."""
    result = []
    for key, value in values.items():
        v = str(value)
        if not re.fullmatch(r'[A-Z][A-Z0-9_]*', key):
            raise ValueError(f'Invalid PBS environment variable name: {key}')
        if any(c in v for c in ',\n\r'):
            raise ValueError(f'Invalid qsub -v value in {key} (comma/newline): {v}')
        result.append(f'{key}={v}')
    return ','.join(result)


def submit(pbs: str, name: str, env: dict, depends: str | None = None,
           dry_run: bool = False, ordinal: int = 1) -> str:
    script = PBS_DIR / pbs
    if not script.exists():
        raise FileNotFoundError(f'Missing PBS script: {script}')
    cmd = ['qsub', '-N', name]
    if depends:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', depends):
            raise ValueError(f'Invalid dependency job ID: {depends}')
        cmd += ['-W', f'depend=afterok:{depends}']
    # Optional explicitly configured Conda bootstrap override must reach the
    # compute node, not just the submission shell. PBS -v is explicit.
    pbs_env = dict(env)
    conda_sh = os.environ.get('FARE_CONDA_SH', '').strip()
    if conda_sh and 'FARE_CONDA_SH' not in pbs_env:
        pbs_env['FARE_CONDA_SH'] = str(Path(conda_sh).expanduser())
    cmd += ['-v', to_qsub_env(pbs_env), str(script)]
    if dry_run:
        print('FARE_GADI_DRY_RUN ' + json.dumps(cmd, ensure_ascii=False), flush=True)
        return f'dryrun-{ordinal}.gadi-pbs'
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True,
                            text=True, check=False)
    if result.returncode:
        raise RuntimeError(f'qsub failed for {pbs}: {result.stderr.strip()} '
                           f'(stdout: {result.stdout.strip()})')
    job_id = result.stdout.strip().splitlines()[-1].strip()
    if not re.fullmatch(r'[0-9]+(?:\.[A-Za-z0-9-]+)?', job_id):
        raise RuntimeError(f'qsub did not return a valid PBS job ID: {result.stdout!r}')
    return job_id


def submit_train(args):
    variant=getattr(args,'variant','full')
    pilot=False
    if args.config:
        import yaml
        with open(Path(args.config).expanduser(),'r',encoding='utf-8') as fh:
            config_for_job=yaml.safe_load(fh)
        pilot=config_for_job.get('experiment',{}).get('run_tag')=='pilot'
    if pilot and args.finalize:
        raise ValueError('Pilot cannot run OOF/finalize; first verify full formal protocol')
    if variant == 'no_memory' and args.finalize:
        raise ValueError('No-Memory baseline has no Retrieval utility for OOF --finalize')
    if args.eval_splits is not None and not args.auto_evaluate:
        raise ValueError('--eval-splits requires --auto-evaluate (omit --no-auto-evaluate)')
    if args.eval_splits is not None and len(set(args.eval_splits)) != len(args.eval_splits):
        raise ValueError('Duplicate entries in --eval-splits')
    splits = list(args.eval_splits) if args.eval_splits is not None else (['val'] if pilot else ['test'])
    if pilot and 'test' in splits and args.auto_evaluate:
        print('FARE_PILOT_TEST_EXPLORATORY: Test is evaluated strictly after training and '
              'checkpoint selection on Validation. Do not use Pilot Test to tune or '
              'select a later Formal model.', flush=True)
    run_stamp = args.run_id or datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    targets = tuple(dict.fromkeys(args.targets))
    if args.backbone_checkpoint and len(targets) != 1:
        raise ValueError('Custom --backbone-checkpoint requires exactly one target; speed/flow use different weights')
    successes = 0
    for target in targets:
        run_dir = (PROJECT_ROOT / 'FARE_exps' / target / args.dataset /
                   ('pilot' if pilot and variant=='full' else 'pilot_no_memory' if pilot else 'formal' if variant=='full' else 'formal_no_memory') / f'{run_stamp}_seed{args.seed}').resolve()
        env = dict(DATASET=args.dataset, TARGET=target, SEED=args.seed,
                   RUN_DIR=str(run_dir),
                   AUTO_EVALUATE=('1' if args.auto_evaluate else '0'),
                   EVAL_SPLITS=':'.join(splits),
                   SEVERITIES=':'.join(map(str,args.severities)),
                   FARE_VARIANT=variant)
        if args.config:
            config_path=Path(args.config).expanduser().resolve()
            if not args.dry_run and not config_path.is_file():raise FileNotFoundError(config_path)
            env['FARE_CONFIG']=str(config_path)
        if args.backbone_checkpoint:
            env['BACKBONE_CHECKPOINT'] = str(Path(args.backbone_checkpoint).resolve())
        jobs = {'dataset': args.dataset, 'target': target, 'seed': args.seed,
                'run_dir': str(run_dir), 'eval_after_train': args.auto_evaluate,
                'evaluation_in_train_allocation': True,
                'eval_splits': splits if args.auto_evaluate else [],
                'severities': list(args.severities),
                'jobs': {}}
        suffix = f'{args.dataset[-2:]}{target[:2]}'
        try:
            train_id = submit('train_formal.pbs', f'FAR{suffix}tr', env,
                              dry_run=args.dry_run, ordinal=successes * 5 + 1)
            jobs['jobs']['train'] = train_id
            print(f'FARE_GADI_TRAIN_SUBMITTED dataset={args.dataset} target={target} '
                  f'job_id={train_id} run_dir={run_dir}', flush=True)
            if args.auto_evaluate:
                print(f'FARE_GADI_IN_JOB_EVALUATE_CONFIGURED dataset={args.dataset} '
                      f'target={target} train_job_id={train_id} '
                      f'splits={":".join(splits)} '
                      f'severities={":".join(map(str,args.severities))} '
                      f'checkpoint={run_dir / "candidate_best.pt"} '
                      f'new_pbs_jobs=0', flush=True)
            if args.finalize:
                oof_id = submit('oof_controller.pbs', f'FAR{suffix}oo',
                                {'CHECKPOINT':str(run_dir / 'candidate_best.pt'),
                                 'DATASET':args.dataset, 'TARGET':target},
                                depends=train_id, dry_run=args.dry_run,
                                ordinal=successes * 5 + 3)
                cal_id = submit('calibrate.pbs', f'FAR{suffix}ca',
                                {'CHECKPOINT':str(run_dir / 'controller_best.pt'),
                                 'DATASET':args.dataset,'TARGET':target},
                                depends=oof_id, dry_run=args.dry_run,
                                ordinal=successes * 5 + 4)
                final_ev = submit('evaluate.pbs', f'FAR{suffix}fe',
                                  {'RUN_DIR':str(run_dir), 'CHECKPOINT_KIND':'best',
                                   'EVAL_KIND':'auto', 'SPLIT':'test',
                                   'SEVERITIES':':'.join(map(str,args.severities)),
                                   'DATASET':args.dataset, 'TARGET':target},
                                  depends=cal_id, dry_run=args.dry_run,
                                  ordinal=successes * 5 + 5)
                jobs['jobs'].update(oof=oof_id, calibration=cal_id,
                                    final_evaluate=final_ev)
                print(f'FARE_GADI_FINAL_CHAIN_SUBMITTED target={target} '
                      f'oof={oof_id} calibration={cal_id} final_evaluate={final_ev}', flush=True)
            if not args.dry_run:
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / 'gadi_jobs.json').write_text(
                    json.dumps(jobs, indent=2, ensure_ascii=False), encoding='utf-8')
            successes += 1
        except Exception as exc:
            print(f'FARE_GADI_SUBMISSION_ERROR dataset={args.dataset} target={target}: '
                  f'{exc}', file=sys.stderr, flush=True)
            if jobs['jobs'] and not args.dry_run:
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / 'gadi_jobs.json').write_text(
                    json.dumps({**jobs, 'error':str(exc)}, indent=2), encoding='utf-8')
            raise
    print(f'FARE_GADI_SUBMISSION_PASS targets={",".join(targets)}', flush=True)


def submit_evaluation(args):
    if not args.checkpoint and not args.run_dir:
        raise ValueError('Provide either --checkpoint or --run-dir')
    if args.checkpoint and args.run_dir:
        ck = Path(args.checkpoint).resolve()
        run_dir = Path(args.run_dir).resolve()
        if ck.parent != run_dir:
            raise ValueError('--checkpoint and --run-dir must refer to the same run')
    else:
        ck = Path(args.checkpoint).resolve() if args.checkpoint else None
        run_dir = (Path(args.run_dir).resolve() if args.run_dir
                   else ck.parent)
    kind = args.checkpoint_kind
    if ck is None:
        basename = ('candidate_best.pt' if kind == 'candidate'
                    else 'best.pt' if kind == 'best' else f'{kind}_best.pt')
        ck = run_dir / basename
    if not args.dry_run and not ck.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {ck}')
    env = dict(CHECKPOINT=str(ck), RUN_DIR=str(run_dir),
               CHECKPOINT_KIND=kind, EVAL_KIND='manual',
               SPLIT=args.split, MODE=args.mode,
               SEVERITIES=':'.join(map(str,args.severities)))
    if args.dataset: env['DATASET'] = args.dataset
    if args.target: env['TARGET'] = args.target
    if args.output: env['EVAL_OUTPUT'] = str(Path(args.output).resolve())
    if args.batch_size: env['BATCH_SIZE'] = args.batch_size
    job_id = submit('evaluate.pbs', 'FAREmanual', env, dry_run=args.dry_run)
    print(f'FARE_GADI_MANUAL_EVALUATE_SUBMITTED job_id={job_id} '
          f'checkpoint={ck} split={args.split} mode={args.mode}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    train = sub.add_parser('train', help='Submit one job/target with same-allocation Evaluate')
    train.add_argument('--dataset', required=True, choices=DATASETS)
    train.add_argument('--targets', nargs='+', choices=['speed','flow'], default=['speed','flow'])
    train.add_argument('--seed', type=int, default=10)
    train.add_argument('--variant',choices=['full','no_memory'],default='full')
    train.add_argument('--run-id', help='Optional shared run stamp')
    train.add_argument('--config',help='Optional YAML shared by speed/flow; architecture stored in checkpoint')
    train.add_argument('--backbone-checkpoint')
    train.add_argument('--severities', nargs='+', type=int, choices=[20,40,60,90],
                       default=[20,40,60,90])
    train.add_argument('--auto-evaluate', dest='auto_evaluate', action='store_true', default=True)
    train.add_argument('--no-auto-evaluate', dest='auto_evaluate', action='store_false')
    train.add_argument('--eval-splits', nargs='+', choices=['val', 'test'],
                       help='Post-train evaluation splits, e.g. --eval-splits val test. '
                            'Default: pilot=val; formal=test. Test is never used for training.')
    train.add_argument('--finalize', action='store_true',
                       help='Also chain OOF -> calibration -> best.pt final Evaluate')
    train.add_argument('--dry-run', action='store_true')
    ev = sub.add_parser('evaluate', help='Submit an independent manual Evaluate')
    ev.add_argument('--checkpoint')
    ev.add_argument('--run-dir')
    ev.add_argument('--checkpoint-kind', choices=PHASES, default='best')
    ev.add_argument('--split', choices=['val','test'], default='test')
    ev.add_argument('--severities', nargs='+', type=int, choices=[20,40,60,90],
                    default=[20,40,60,90])
    ev.add_argument('--mode', default='full', choices=['full','retrieval_off','state_only',
                         'innovation_only','global_retrieval','no_mask_alignment',
                         'random_memory','no_risk','no_utility','constant_alpha','no_memory','no_learned_reranker','no_observation_distance'])
    ev.add_argument('--dataset', choices=DATASETS)
    ev.add_argument('--target', choices=['speed','flow'])
    ev.add_argument('--batch-size', type=int)
    ev.add_argument('--output')
    ev.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.action == 'train': submit_train(args)
    else: submit_evaluation(args)

if __name__ == '__main__':
    main()
