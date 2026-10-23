#!/usr/bin/env python3
"""Submit FARE v6.2 Experiment 1 GPU stages to Gadi PBS.

No previous FARE checkpoint input is accepted. The only prior artifacts are
original STLAformer and paired OOF. Uses submitter's exact Python executable.
"""
import _bootstrap
import argparse,sys,json,subprocess
from pathlib import Path
from fare.config import read_config,root,project_path,backbone_checkpoint
from fare.experiment1_v62 import VARIANTS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',required=True,choices=('speed','flow'))
    p.add_argument('--stage',required=True,choices=('all','foundation','feasibility','transfer','assess'))
    p.add_argument('--variant',choices=VARIANTS)
    p.add_argument('--dataset',default='TraffiDent_Net_06')
    p.add_argument('--seed',type=int,default=10)
    p.add_argument('--config',default='FARE/configs/experiment1.yaml')
    p.add_argument('--backbone-checkpoint')
    p.add_argument('--memory')
    p.add_argument('--experiment-dir')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    if args.stage=='transfer' and not args.variant:raise ValueError('--variant required for transfer')
    cfg=read_config(args.config,args.dataset,args.target,args.seed)
    out=(project_path(args.experiment_dir) if args.experiment_dir else
         root()/'FARE_exps'/args.target/args.dataset/'v62_experiment1'/(
             f'seed{args.seed}'+('_smoke' if args.smoke else '')))
    summary_file=out/'support'/'support_summary.json'
    summary=json.loads(summary_file.read_text()) if summary_file.exists() else {}
    if summary and args.backbone_checkpoint and summary.get('backbone_checkpoint'):
        if project_path(args.backbone_checkpoint,must_exist=True)!=project_path(summary['backbone_checkpoint'],must_exist=True):
            raise ValueError('Explicit Backbone conflicts with frozen foundation Backbone')
    if summary and args.memory and summary.get('memory_path'):
        if project_path(args.memory,must_exist=True)!=project_path(summary['memory_path'],must_exist=True):
            raise ValueError('Explicit OOF Memory conflicts with foundation Memory')
    saved_backbone=summary.get('backbone_checkpoint')
    saved_memory=summary.get('memory_path')
    backbone=backbone_checkpoint(cfg,args.backbone_checkpoint or saved_backbone)
    memory=project_path(args.memory or saved_memory,must_exist=True) if (args.memory or saved_memory) else root()/'FARE_cache'/args.target/args.dataset/f'memory_v502_{args.seed}_{cfg["memory"]["max_episodes"]}'
    if not (memory/'manifest.json').exists():raise FileNotFoundError(memory/'manifest.json')
    if args.stage=='assess':
        cmd=[sys.executable,'-u',str(root()/'FARE/scripts/experiment1.py'),
             '--stage','assess','--target',args.target,'--dataset',args.dataset,
             '--seed',str(args.seed),'--config',str(project_path(args.config,must_exist=True)),
             '--experiment-dir',str(out)]
        print('FARE62_E1_ASSESS '+str(out),flush=True)
        if not args.check_only:subprocess.run(cmd,check=True)
        return
    env={'SPEED_LLM_RAG_ROOT':str(root()),'FARE_PYTHON':str(Path(sys.executable).resolve()),
         'TARGET':args.target,'FARE62_STAGE':args.stage,'FARE62_DATASET':args.dataset,
         'FARE62_SEED':str(args.seed),'FARE62_CONFIG':str(project_path(args.config,must_exist=True)),
         'BACKBONE_CHECKPOINT':str(backbone),'FARE_MEMORY':str(memory),
         'FARE62_EXPERIMENT_DIR':str(out),'FARE62_SMOKE':'1' if args.smoke else '0'}
    if args.variant:env['FARE62_VARIANT']=args.variant
    if any(',' in v or '\n' in v for v in env.values()):raise ValueError('Unsafe PBS environment value')
    print('FARE62_E1_PRE_SUBMIT_PASS '+json.dumps({'stage':args.stage,'variant':args.variant,
        'target':args.target,'experiment_dir':str(out),'backbone':str(backbone),
        'memory':str(memory),'python':env['FARE_PYTHON'],'smoke':args.smoke}),flush=True)
    if args.check_only:return
    if args.stage=='foundation' and (out/'support'/'support_best.pt').exists():
        raise FileExistsError('Support checkpoint exists. Never overwrite a fresh experiment foundation.')
    if args.stage=='transfer':
        if not (out/'support'/'support_best.pt').exists() or not (out/'candidate_bank'/'manifest.json').exists():
            raise FileNotFoundError('Complete foundation before submitting Transfer variants')
        if (out/'transfer'/args.variant/'best.pt').exists():raise FileExistsError('Variant already completed')
    script=root()/'FARE/pbs/experiment1_v62.pbs'
    job=subprocess.check_output(['qsub','-v',','.join(f'{k}={v}' for k,v in env.items()),str(script)],text=True).strip()
    print(f'FARE62_E1_JOB={job}\nFARE62_E1_OUTPUT={out}',flush=True)

if __name__=='__main__':main()
