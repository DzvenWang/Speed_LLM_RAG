#!/usr/bin/env python3
import argparse,subprocess,sys
from pathlib import Path
import _bootstrap
from fare.config import load_config,config_from_checkpoint,json_write
from fare.runtime import setup,make_run_dir
from fare.trainer import FARETrainer,seed_all

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--dataset',default=None);a.add_argument('--target',choices=['speed','flow'],default=None)
    a.add_argument('--seed',type=int,default=None);a.add_argument('--config');a.add_argument('--backbone-checkpoint');a.add_argument('--memory')
    a.add_argument('--variant',choices=['full','no_memory'],default='full');a.add_argument('--smoke',action='store_true');a.add_argument('--run-dir');a.add_argument('--resume');a.add_argument('--phases',nargs='+',default=None)
    a.add_argument('--auto-evaluate',action='store_true');a.add_argument('--batch-size',type=int);a.add_argument('--num-workers',type=int)
    args=a.parse_args()
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
    trainer=FARETrainer(model,memory,train,val,cfg,meta,device,run_dir,memory_path)
    best=trainer.train(args.phases,args.smoke,args.resume)
    print(f'FARE_TRAIN_PASS run_dir={run_dir} candidate_checkpoint={best}',flush=True)
    if args.auto_evaluate and not args.smoke:
        # Direct qsub fallback. For Gadi, prefer submit_gadi.py so Evaluate
        # runs in its own afterok PBS job rather than consuming train walltime.
        print(f'FARE_AUTO_EVALUATE_START checkpoint={best}',flush=True)
        split='val' if cfg['experiment'].get('run_tag')=='pilot' else 'test'
        cmd=[sys.executable,str(Path(__file__).with_name('evaluate.py')),
             '--checkpoint',str(best),'--dataset',args.dataset,
             '--target',args.target,'--memory',str(memory_path),
             '--split',split,'--output',str(run_dir/'evaluation_candidate')]
        if split=='test':cmd+=['--severities','20','40','60','90']
        subprocess.check_call(cmd)
        print(f'FARE_AUTO_EVALUATE_PASS checkpoint={best}',flush=True)
if __name__=='__main__':main()
