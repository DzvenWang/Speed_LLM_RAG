#!/usr/bin/env python3
"""Submit ONE PBS job per FARE target: train -> evaluate -> internal analysis.

All three phases run sequentially on the same GPU allocation.
No dependent second GPU job and no retrained ablation jobs.
"""
import _bootstrap,argparse,subprocess,time
from pathlib import Path
from fare.config import read_config,root

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='FARE/configs/pilot.yaml')
    p.add_argument('--target',required=True,choices=['speed','flow'])
    p.add_argument('--dataset',default='TraffiDent_Net_06')
    p.add_argument('--seed',type=int,default=10)
    p.add_argument('--checkpoint')
    p.add_argument('--oos-forecast-file')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--run-dir')
    g=p.add_mutually_exclusive_group()
    g.add_argument('--eval-val',action='store_true',help='Also evaluate/analyze Val after training')
    g.add_argument('--test-only',action='store_true',help='Only Test s20/s40/s60/s90 after training')
    a=p.parse_args()
    cfg=read_config(a.config,a.dataset,a.target,a.seed,a.smoke)
    if a.eval_val:cfg['evaluation']['include_val']=True
    if a.test_only:cfg['evaluation']['include_val']=False
    run_dir=(Path(a.run_dir).expanduser().resolve() if a.run_dir else
             root()/'FARE_exps'/a.target/a.dataset/cfg['experiment']['run_tag']/f'{time.strftime("%Y%m%d_%H%M%S")}_seed{a.seed}')
    run_dir.mkdir(parents=True,exist_ok=True)
    # qsub -v explicitly propagates only the necessary variables to the compute node.
    env={'TARGET':a.target,'DATASET':a.dataset,'SEED':str(a.seed),
         'FARE_CONFIG':str(Path(a.config).resolve()),'FARE_RUN_DIR':str(run_dir),
         'FARE_SMOKE':'1' if a.smoke else '0',
         'FARE_EVAL_VAL':'1' if cfg['evaluation']['include_val'] else '0',
         'SPEED_LLM_RAG_ROOT':str(root())}
    if a.checkpoint:env['BACKBONE_CHECKPOINT']=str(Path(a.checkpoint).resolve())
    if a.oos_forecast_file:env['FARE_OOS_FORECAST_FILE']=str(Path(a.oos_forecast_file).resolve())
    script=root()/'FARE/pbs/full_pipeline.pbs'
    job=subprocess.check_output(['qsub','-v',','.join(f'{k}={v}' for k,v in env.items()),str(script)],text=True).strip()
    print(f'FARE3_FULL_PIPELINE_JOB={job}\nFARE3_RUN_DIR={run_dir}\nFARE3_POST_TRAINING_EVALUATION={"val+test" if cfg["evaluation"]["include_val"] else "test_only"}',flush=True)

if __name__=='__main__':main()
