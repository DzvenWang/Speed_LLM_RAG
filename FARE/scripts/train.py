#!/usr/bin/env python3
import argparse,gc,json,subprocess,sys,os
from pathlib import Path
import _bootstrap
import torch
from fare.config import load_config,config_from_checkpoint,json_write
from fare.runtime import setup,make_run_dir
from fare.trainer import FARETrainer,seed_all

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--dataset',default=None);a.add_argument('--target',choices=['speed','flow'],default=None)
    a.add_argument('--seed',type=int,default=None);a.add_argument('--config');a.add_argument('--backbone-checkpoint');a.add_argument('--memory')
    a.add_argument('--variant',choices=['full','no_memory'],default='full');a.add_argument('--smoke',action='store_true');a.add_argument('--run-dir');a.add_argument('--resume');a.add_argument('--phases',nargs='+',default=None)
    a.add_argument('--auto-evaluate',action='store_true');a.add_argument('--batch-size',type=int);a.add_argument('--num-workers',type=int)
    a.add_argument('--diag-interval',type=int,default=None,help='Sample every N training/validation batches')
    a.add_argument('--diag-max-batches',type=int,default=None,help='Maximum diagnostic samples per epoch (0=unlimited)')
    a.add_argument('--progress-every',type=int,default=None,help='Log progress heartbeat every N batches (0=off)')
    a.add_argument('--no-online-diagnostics',action='store_true')
    a.add_argument('--eval-splits',nargs='+',choices=['val','test'],default=None,
                   help='Post-train evaluation split(s). Default: pilot=val, formal=test.')
    a.add_argument('--eval-severities',nargs='+',type=int,choices=[20,40,60,90],default=[20,40,60,90])
    args=a.parse_args()
    if args.eval_splits and not args.auto_evaluate:
        raise ValueError('--eval-splits requires --auto-evaluate')
    if args.eval_splits and len(set(args.eval_splits))!=len(args.eval_splits):
        raise ValueError('Duplicate --eval-splits are not permitted')
    if args.eval_severities and len(set(args.eval_severities))!=len(args.eval_severities):
        raise ValueError('Duplicate --eval-severities are not permitted')
    if args.resume:
        from fare.trainer import load_ckpt
        prior=load_ckpt(args.resume)
        cfg=config_from_checkpoint(prior,smoke=args.smoke)
        if args.config:raise ValueError('Resume uses checkpoint configuration, not a new YAML')
        for key in ('dataset','target','seed'):
            requested=getattr(args,key)
            if requested is not None and requested!=cfg['experiment'][key]:
                raise ValueError(f'Resume {key} mismatch: {requested} vs {cfg["experiment"][key]}')
    else:
        cfg=load_config(args.config,args.dataset or 'TraffiDent_Net_06',
                        args.target or 'speed',args.seed if args.seed is not None else 10,args.smoke)
    if args.resume and args.variant != cfg['experiment'].get('training_mode','full'):
        raise ValueError('Resume --variant must match checkpoint training mode')
    cfg['experiment']['training_mode']=args.variant
    args.dataset=cfg['experiment']['dataset']
    args.target=cfg['experiment']['target']
    args.seed=cfg['experiment']['seed']
    if args.batch_size:cfg['training']['batch_size']=args.batch_size
    if args.num_workers is not None:cfg['training']['num_workers']=args.num_workers
    diag=cfg.setdefault('diagnostics',{})
    if args.diag_interval is not None:diag['sample_every_batches']=args.diag_interval
    if args.diag_max_batches is not None:diag['max_sampled_batches']=args.diag_max_batches
    if args.progress_every is not None:diag['progress_every_batches']=args.progress_every
    if args.no_online_diagnostics:diag['enabled']=False
    seed_all(args.seed)
    backbone_to_use=args.backbone_checkpoint
    memory_to_use=args.memory
    if args.resume:
        saved_backbone=prior.get('metadata',{}).get('backbone_checkpoint')
        saved_memory=prior.get('memory_path')
        if backbone_to_use and saved_backbone and Path(backbone_to_use).resolve()!=Path(saved_backbone).resolve():
            raise ValueError('Resume backbone differs from checkpoint backbone')
        if memory_to_use and saved_memory and Path(memory_to_use).resolve()!=Path(saved_memory).resolve():
            raise ValueError('Resume memory differs from checkpoint memory')
        backbone_to_use=backbone_to_use or saved_backbone
        memory_to_use=memory_to_use or saved_memory
    model,memory,train,val,device,meta=setup(cfg,backbone_to_use,memory_to_use,smoke=args.smoke)
    if args.resume and meta['backbone_sha256']!=prior['metadata']['backbone_sha256']:
        raise RuntimeError('Resume backbone SHA256 mismatch')
    run_dir=Path(args.run_dir) if args.run_dir else (Path(args.resume).resolve().parent if args.resume else make_run_dir(cfg,'formal_smoke' if args.smoke else cfg['experiment'].get('run_tag','formal')+('_no_memory' if args.variant=='no_memory' else '')))
    memory_path=run_dir/'memory.npz'
    if not memory_to_use:
        memory.save(memory_path,{'dataset':args.dataset,'target':args.target,'train_only':True})
    else:
        memory_path=Path(memory_to_use)
    if args.variant=='no_memory' and args.phases is None:
        args.phases=['injection','partial','full']
    print('FARE_DIAGNOSTICS_CONFIG '+json.dumps({'run_dir':str(run_dir),
         'interval':diag.get('sample_every_batches',50),
         'max_batches':diag.get('max_sampled_batches',160),
         'resource_monitor':os.environ.get('FARE_RESOURCE_OUTPUT_PREFIX'),
         'enabled':diag.get('enabled',True)}),flush=True)
    trainer=FARETrainer(model,memory,train,val,cfg,meta,device,run_dir,memory_path)
    best=trainer.train(args.phases,args.smoke,args.resume)
    print(f'FARE_TRAIN_PASS run_dir={run_dir} candidate_checkpoint={best}',flush=True)
    if args.auto_evaluate and not args.smoke:
        # The next subprocess remains in the CURRENT PBS allocation / GPU.
        # Free the parent's training CUDA buffers before starting evaluation.
        splits=args.eval_splits or (['val'] if cfg['experiment'].get('run_tag')=='pilot' else ['test'])
        if cfg['experiment'].get('run_tag')=='pilot' and 'test' in splits:
            print('FARE_PILOT_TEST_EXPLORATORY: Validation selects the checkpoint; '
                  'Pilot Test is not an unbiased final comparison if used for tuning.',flush=True)
        del trainer,model,memory,train,val
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        report_path=run_dir/'posttrain_evaluation.json'
        results={'checkpoint':str(best),'run_tag':cfg['experiment'].get('run_tag','formal'),
                 'same_allocation':True,'requested_splits':splits,'completed_splits':[],
                 'severities':args.eval_severities,'status':'running',
                 'pilot_test_exploratory':bool(cfg['experiment'].get('run_tag')=='pilot' and 'test' in splits)}
        json_write(report_path,results)
        print(f'FARE_IN_JOB_EVALUATION_START checkpoint={best} splits={":".join(splits)} '
              f'severities={":".join(map(str,args.eval_severities))} '
              f'pbs_job_id={__import__("os").environ.get("PBS_JOBID","local")}',flush=True)
        try:
            for split in splits:
                cmd=[sys.executable,'-u',str(Path(__file__).with_name('evaluate.py')),
                     '--checkpoint',str(best),'--dataset',args.dataset,
                     '--target',args.target,'--memory',str(memory_path),
                     '--split',split,'--output',str(run_dir/'evaluation_candidate')]
                if split=='test':cmd+=['--severities',*map(str,args.eval_severities)]
                print(f'FARE_IN_JOB_EVALUATION_SPLIT_START split={split}',flush=True)
                subprocess.run(cmd,check=True)
                results['completed_splits'].append(split)
                json_write(report_path,results)
                print(f'FARE_IN_JOB_EVALUATION_SPLIT_PASS split={split}',flush=True)
        except Exception as exc:
            results['status']='failed'
            results['error']=str(exc)
            json_write(report_path,results)
            print(f'FARE_IN_JOB_EVALUATION_FAIL error={exc}',flush=True)
            raise
        results['status']='pass'
        json_write(report_path,results)
        print(f'FARE_IN_JOB_EVALUATION_PASS completed={":".join(results["completed_splits"])}',flush=True)
        print(f'FARE_AUTO_EVALUATE_PASS checkpoint={best}',flush=True)
if __name__=='__main__':main()
