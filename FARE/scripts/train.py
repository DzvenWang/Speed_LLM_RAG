#!/usr/bin/env python3
import argparse,subprocess,sys
from pathlib import Path
import _bootstrap
from fare.config import load_config,json_write
from fare.runtime import setup,make_run_dir
from fare.trainer import FARETrainer,seed_all

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--dataset',default='TraffiDent_Net_06');a.add_argument('--target',choices=['speed','flow'],default='speed')
    a.add_argument('--seed',type=int,default=10);a.add_argument('--config');a.add_argument('--backbone-checkpoint');a.add_argument('--memory')
    a.add_argument('--smoke',action='store_true');a.add_argument('--run-dir');a.add_argument('--resume');a.add_argument('--phases',nargs='+',default=None)
    a.add_argument('--auto-evaluate',action='store_true');a.add_argument('--batch-size',type=int);a.add_argument('--num-workers',type=int)
    args=a.parse_args();cfg=load_config(args.config,args.dataset,args.target,args.seed,args.smoke)
    if args.batch_size:cfg['training']['batch_size']=args.batch_size
    if args.num_workers is not None:cfg['training']['num_workers']=args.num_workers
    seed_all(args.seed)
    model,memory,train,val,device,meta=setup(cfg,args.backbone_checkpoint,args.memory,smoke=args.smoke)
    run_dir=Path(args.run_dir) if args.run_dir else (Path(args.resume).resolve().parent if args.resume else make_run_dir(cfg,'formal_smoke' if args.smoke else 'formal'))
    memory_path=run_dir/'memory.npz'
    if not args.memory:memory.save(memory_path,{'dataset':args.dataset,'target':args.target,'train_only':True})
    else:memory_path=Path(args.memory)
    trainer=FARETrainer(model,memory,train,val,cfg,meta,device,run_dir,memory_path)
    best=trainer.train(args.phases,args.smoke,args.resume)
    print(f'FARE_TRAIN_PASS run_dir={run_dir} candidate_checkpoint={best}',flush=True)
    if args.auto_evaluate and not args.smoke:
        # Direct qsub fallback. For Gadi, prefer submit_gadi.py so Evaluate
        # runs in its own afterok PBS job rather than consuming train walltime.
        print(f'FARE_AUTO_EVALUATE_START checkpoint={best}',flush=True)
        cmd=[sys.executable,str(Path(__file__).with_name('evaluate.py')),
             '--checkpoint',str(best),'--dataset',args.dataset,
             '--target',args.target,'--memory',str(memory_path),
             '--split','test','--severities','20','40','60','90',
             '--output',str(run_dir/'evaluation_candidate')]
        subprocess.check_call(cmd)
        print(f'FARE_AUTO_EVALUATE_PASS checkpoint={best}',flush=True)
if __name__=='__main__':main()
