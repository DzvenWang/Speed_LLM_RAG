#!/usr/bin/env python3
"""Submit ONE PBS job per FARE target: train -> evaluate -> internal analysis.

All three phases run sequentially on the same GPU allocation.
No dependent second GPU job and no retrained ablation jobs.
"""
import _bootstrap,argparse,subprocess,time,sys,os
from pathlib import Path
from fare.config import read_config,root,project_path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='FARE/configs/formal.yaml')
    p.add_argument('--target',required=True,choices=['speed','flow'])
    p.add_argument('--dataset',default='TraffiDent_Net_06')
    p.add_argument('--seed',type=int,default=10)
    p.add_argument('--checkpoint')
    p.add_argument('--oos-forecast-file')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--run-dir')
    p.add_argument('--memory',help='Prebuilt v5 Memory directory, relative to repo or absolute')
    p.add_argument('--resume',help='Existing v6 last.pt, relative to repo or absolute')
    p.add_argument('--evaluate-only',action='store_true')
    p.add_argument('--skip-analysis',action='store_true')
    p.add_argument('--skip-precheck',action='store_true',help='Bypass CPU safety gate (not recommended)')
    p.add_argument('--check-only',action='store_true',help='Run the full CPU safety gate without submitting PBS')
    g=p.add_mutually_exclusive_group()
    g.add_argument('--eval-val',action='store_true',help='Also evaluate/analyze Val after training')
    g.add_argument('--test-only',action='store_true',help='Only Test s20/s40/s60/s90 after training')
    a=p.parse_args()
    cfg=read_config(a.config,a.dataset,a.target,a.seed,a.smoke)
    if a.eval_val:cfg['evaluation']['include_val']=True
    if a.test_only:cfg['evaluation']['include_val']=False
    if not a.skip_precheck:
        checker=root()/'FARE/scripts/pre_submit_check.py'
        if not checker.is_file():raise FileNotFoundError(checker)
        check_cmd=[sys.executable,'-u',str(checker),'--config',str(project_path(a.config,must_exist=True)),
                   '--dataset',a.dataset,'--target',a.target,'--seed',str(a.seed)]
        if a.checkpoint:check_cmd+=['--checkpoint',str(project_path(a.checkpoint,must_exist=True))]
        if a.smoke:check_cmd+=['--smoke']
        print('FARE_PRE_SUBMIT_GATE_START',flush=True)
        subprocess.check_call(check_cmd)
        preflight=[sys.executable,'-u',str(root()/'FARE/scripts/preflight.py'),
                   '--config',str(project_path(a.config,must_exist=True)),'--dataset',a.dataset,'--target',a.target]
        if a.checkpoint:preflight+=['--checkpoint',str(project_path(a.checkpoint,must_exist=True))]
        if a.oos_forecast_file:preflight+=['--oos-forecast-file',str(project_path(a.oos_forecast_file,must_exist=True))]
        if a.memory:preflight+=['--memory',str(project_path(a.memory,must_exist=True))]
        subprocess.check_call(preflight)
        print('FARE_PRE_SUBMIT_GATE_PASS',flush=True)
    if a.check_only:
        print('FARE_CHECK_ONLY_NO_QSUB',flush=True)
        return
    run_dir=(project_path(a.run_dir) if a.run_dir else
             root()/'FARE_exps'/a.target/a.dataset/cfg['experiment']['run_tag']/f'{time.strftime("%Y%m%d_%H%M%S")}_seed{a.seed}')
    run_dir.mkdir(parents=True,exist_ok=True)
    # qsub -v explicitly propagates only the necessary variables to the compute node.
    env={'TARGET':a.target,'DATASET':a.dataset,'SEED':str(a.seed),
         'FARE_CONFIG':str(project_path(a.config,must_exist=True)),'FARE_RUN_DIR':str(run_dir),
         'FARE_SMOKE':'1' if a.smoke else '0',
         'FARE_EVAL_VAL':'1' if cfg['evaluation']['include_val'] else '0',
         'SPEED_LLM_RAG_ROOT':str(root()),
         'FARE_PYTHON':str(Path(sys.executable).absolute()),
         'FARE_ENV_PREFIX':str(Path(sys.prefix).resolve()),
         'FARE_EVALUATE_ONLY':'1' if a.evaluate_only else '0',
         'FARE_SKIP_ANALYSIS':'1' if a.skip_analysis else '0'}
    if a.checkpoint:env['BACKBONE_CHECKPOINT']=str(project_path(a.checkpoint,must_exist=True))
    if a.oos_forecast_file:env['FARE_OOS_FORECAST_FILE']=str(project_path(a.oos_forecast_file,must_exist=True))
    if a.memory:env['FARE_MEMORY']=str(project_path(a.memory,must_exist=True))
    if a.resume:env['FARE_RESUME']=str(project_path(a.resume,must_exist=True))
    if cfg['training'].get('initialize_from'):
        raise ValueError('FARE requires fresh initialization; remove training.initialize_from')
    script=root()/'FARE/pbs/full_pipeline.pbs'
    if not script.is_file():raise FileNotFoundError(f'Missing PBS script: {script}')
    if not Path(sys.executable).is_file():raise FileNotFoundError(f'Missing active Python: {sys.executable}')
    if any(',' in v or '\n' in v for v in env.values()):
        raise ValueError('PBS -v does not support commas or newlines in path values')
    print(f'FARE6_SUBMIT_ENV python={env["FARE_PYTHON"]} prefix={env["FARE_ENV_PREFIX"]} root={root()}',flush=True)
    job=subprocess.check_output(['qsub','-v',','.join(f'{k}={v}' for k,v in env.items()),str(script)],text=True).strip()
    print(f'FARE6_FULL_PIPELINE_JOB={job}\nFARE5_RUN_DIR={run_dir}\nFARE5_POST_TRAINING_EVALUATION={"val+test" if cfg["evaluation"]["include_val"] else "test_only"}',flush=True)

if __name__=='__main__':main()
