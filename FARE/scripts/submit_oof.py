#!/usr/bin/env python3
"""Submit parallel OOF producer jobs and PBS-dependent finalization/FARE training.

Parallel: one V100 per (target, fold), using independent PBS jobs.
Dependency graph: producers --afterok--> finalize --afterok--> FARE target jobs.
Use --plan-only to validate/cache fold manifests without submitting anything.
"""
import _bootstrap
import argparse,subprocess,os,sys,time,json
from pathlib import Path
from fare.config import root,project_path,read_config,backbone_checkpoint,dump_json


def submit(script,env,dependency=None,job_name=None):
    if any(',' in str(v) or '\\n' in str(v) for v in env.values()):
        raise ValueError('PBS -v cannot encode commas/newlines')
    cmd=['qsub']
    if job_name:cmd+=['-N',job_name]
    if dependency:cmd+=['-W','depend=afterok:'+':'.join(dependency)]
    cmd+=['-v',','.join(f'{k}={v}' for k,v in env.items()),str(script)]
    return subprocess.check_output(cmd,text=True).strip()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',choices=['speed','flow','both'],default='both')
    p.add_argument('--dataset',default='TraffiDent_Net_06')
    p.add_argument('--config',default='FARE/configs/formal.yaml')
    p.add_argument('--nfolds',type=int,default=2)
    p.add_argument('--warmup',type=float,default=.30)
    p.add_argument('--purge',type=int,default=12)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--workers',type=int,default=4,help='DataLoader workers per independent V100 fold (default 4)')
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--seed',type=int,default=10)
    p.add_argument('--output-dir',default='FARE_cache/oof_v503')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--parallel-folds',action='store_true',help='1 GPU job for each target+fold')
    p.add_argument('--auto-fare',action='store_true',help='Also submit dependent FARE formal training/evaluation jobs')
    p.add_argument('--speed-checkpoint')
    p.add_argument('--flow-checkpoint')
    a=p.parse_args()
    if a.auto_fare and not a.parallel_folds:p.error('--auto-fare requires --parallel-folds')
    if a.nfolds<1 or a.epochs<1 or a.batch_size<1 or a.workers<0 or not(0<a.warmup<1):p.error('Invalid parameters')
    absolute_config=str(project_path(a.config,must_exist=True))
    output_dir=str(project_path(a.output_dir))
    cmd=[sys.executable,'-u',str(root()/'FARE/scripts/run_oof_pipeline.py'),
         '--target',a.target,'--dataset',a.dataset,'--config',absolute_config,
         '--nfolds',str(a.nfolds),'--warmup',str(a.warmup),'--purge',str(a.purge),
         '--batch-size',str(a.batch_size),'--workers',str(a.workers),'--epochs',str(a.epochs),'--output-dir',output_dir,'--plan-only',
         '--seed',str(a.seed)]
    if a.speed_checkpoint:cmd+=['--speed-checkpoint',a.speed_checkpoint]
    if a.flow_checkpoint:cmd+=['--flow-checkpoint',a.flow_checkpoint]
    subprocess.check_call(cmd)
    if a.plan_only:
        print('FARE_OOF_PLAN_ONLY_NO_QSUB',flush=True)
        return
    env={'SPEED_LLM_RAG_ROOT':str(root()),'FARE_PYTHON':str(Path(sys.executable).resolve()),
         'FARE_CONFIG':absolute_config,'DATASET':a.dataset,'TARGET':a.target,'SEED':str(a.seed),
         'OOF_FOLDS':str(a.nfolds),'OOF_WARMUP':str(a.warmup),'OOF_PURGE':str(a.purge),
         'OOF_BATCH_SIZE':str(a.batch_size),'OOF_WORKERS':str(a.workers),'OOF_EPOCHS':str(a.epochs),
         'OOF_OUTPUT_DIR':output_dir}
    if a.speed_checkpoint:env['SPEED_CHECKPOINT']=str(project_path(a.speed_checkpoint,must_exist=True))
    if a.flow_checkpoint:env['FLOW_CHECKPOINT']=str(project_path(a.flow_checkpoint,must_exist=True))
    targets=['speed','flow'] if a.target=='both' else [a.target]
    if not a.parallel_folds:
        job=submit(root()/'FARE/pbs/oof_pipeline.pbs',env,job_name='FARE503_OOF')
        print('FARE_OOF_JOB='+job,flush=True)
        return
    stamp=time.strftime('%Y%m%d_%H%M%S')
    receipt=root()/'FARE_exps'/'oof_submissions'/f'{stamp}_seed{a.seed}.json'
    receipt.parent.mkdir(parents=True,exist_ok=True)
    graph={'version':'5.0.3','status':'submitting','targets':targets,'nfolds':a.nfolds,
           'config':absolute_config,'output_dir':output_dir,'jobs':{'producers':{},'finalize':None,'fare':{}},
           'plan':{'warmup':a.warmup,'purge':a.purge,'epochs':a.epochs,'batch_size':a.batch_size,'workers':a.workers}}
    def save():dump_json(receipt,graph)
    save()
    try:
        for t in targets:
            for i in range(a.nfolds):
                vars=dict(env,TARGET=t,OOF_FOLD_ID=str(i))
                job=submit(root()/'FARE/pbs/oof_fold.pbs',vars,job_name=f'OOF_{t[0].upper()}{i}')
                graph['jobs']['producers'][f'{t}_fold{i}']=job;save()
                print(f'FARE_OOF_PRODUCER_JOB target={t} fold={i} job={job}',flush=True)
        preds=list(graph['jobs']['producers'].values())
        graph['jobs']['finalize']=submit(root()/'FARE/pbs/oof_finalize.pbs',env,dependency=preds,job_name='FARE503_OOF_FINAL')
        save()
        print(f'FARE_OOF_FINALIZE_JOB={graph["jobs"]["finalize"]}',flush=True)
        if a.auto_fare:
            for t in targets:
                cfg=read_config(absolute_config,a.dataset,t,seed=a.seed)
                ref=backbone_checkpoint(cfg,getattr(a,f'{t}_checkpoint'))
                run_dir=root()/'FARE_exps'/t/a.dataset/cfg['experiment']['run_tag']/f'{stamp}_seed{a.seed}_after_oof'
                vars={'TARGET':t,'DATASET':a.dataset,'SEED':str(a.seed),
                      'FARE_CONFIG':absolute_config,'FARE_RUN_DIR':str(run_dir),
                      'FARE_SMOKE':'0','FARE_EVAL_VAL':'1' if cfg['evaluation']['include_val'] else '0',
                      'SPEED_LLM_RAG_ROOT':str(root()),'FARE_PYTHON':str(Path(sys.executable).resolve()),
                      'FARE_ENV_PREFIX':str(Path(sys.prefix).resolve()),
                      'FARE_EVALUATE_ONLY':'0','FARE_SKIP_ANALYSIS':'0',
                      'BACKBONE_CHECKPOINT':str(ref),
                      'FARE_OOS_FORECAST_FILE':str(root()/'FARE_cache'/f'oof_{t}_paired.npz')}
                job=submit(root()/'FARE/pbs/full_pipeline.pbs',vars,
                           dependency=[graph['jobs']['finalize']],job_name=f'FARE5_{t}')
                graph['jobs']['fare'][t]=job;save()
                print(f'FARE_FORMAL_AFTER_OOF target={t} job={job} run_dir={run_dir}',flush=True)
        graph['status']='submitted';save()
        print('FARE503_SUBMISSION_RECEIPT='+str(receipt),flush=True)
        print('FARE503_SUBMITTED='+json.dumps(graph['jobs'],sort_keys=True),flush=True)
    except Exception:
        graph['status']='partial_submission_error';save()
        print('WARNING: Partial submission; inspect receipt to avoid duplicate jobs: '+str(receipt),file=sys.stderr,flush=True)
        raise

if __name__=='__main__':main()
