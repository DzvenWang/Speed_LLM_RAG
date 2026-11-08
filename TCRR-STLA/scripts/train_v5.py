#!/usr/bin/env python3
"""Fresh v5 training: S0 -> fresh retrieval/replay -> switch-aware C12 -> C3."""
from __future__ import annotations
import os,sys,datetime,shutil,subprocess,argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tcrr_stla.config import load_config,cache_path,resolve_stla_checkpoint,file_sha256
from tcrr_stla.v5.trainer import fit_shared_adapter,train_v5,seed_all
from tcrr_stla.v5.model import THREE_SWITCH_MODES
from scripts.build_retrieval_cache import ensure_cache
import torch

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--target',required=True,choices=['speed','flow'])
    p.add_argument('--dataset',default='TraffiDent_Net_06')
    p.add_argument('--seed',type=int,default=10)
    p.add_argument('--backbone-checkpoint',required=True)
    p.add_argument('--config',default=str(ROOT/'configs/tcrr_stla_v5.yaml'))
    p.add_argument('--batch-size',type=int)
    p.add_argument('--run-stamp')
    p.add_argument('--resume',help='existing v5 last.pt checkpoint')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--smoke-full-batch',action='store_true',
                   help='Smoke one true-size train batch, repeating isolated smoke query indices as needed to measure peak memory. Requires --smoke and --batch-size.')
    p.add_argument('--skip-evaluate',action='store_true',help='Skip four-severity TEST evaluation (does not skip validation diagnostics)')
    p.add_argument('--skip-diagnostics',action='store_true',help='Disable automatic validation mechanism diagnostics')
    p.add_argument('--diagnostic-max-batches',type=int,default=None,help='Number of batches for each negative control')
    p.add_argument('--diagnostic-primary-max-batches',type=int,default=None,help='Limit primary four/eight-mode validation; 0 = full validation')
    a=p.parse_args()
    if a.batch_size is not None and not (1 <= a.batch_size <= 512):
        p.error('--batch-size must be between 1 and 512')
    if a.smoke_full_batch and (not a.smoke or a.batch_size is None):
        p.error('--smoke-full-batch requires both --smoke and --batch-size')
    os.environ['TRAFFIDENT_TARGET']=a.target
    cfg=load_config(a.config)
    if cfg['version']!='5.3.0':raise ValueError('Expected 5.3.0 config')
    cfg['experiment'].update(dataset=a.dataset,target=a.target,seed=a.seed)
    cfg['training']['ablation']='full' # identical retrieval for all four internal modes
    cfg['runtime']['smoke_full_batch']=bool(a.smoke_full_batch)
    if a.batch_size is not None:cfg['training']['batch_size']=a.batch_size
    if a.smoke_full_batch:
        print(f'V5_SMOKE_FULL_BATCH requested={a.batch_size} unique_smoke_queries=24 '
              'train_batches=1 (duplicated indices only for memory profiling)',flush=True)
    if 'TCRR_TRAIN_WORKERS' in os.environ:
        cfg['runtime']['num_workers']=int(os.environ['TCRR_TRAIN_WORKERS'])
        if cfg['runtime']['num_workers']<0 or cfg['runtime']['num_workers']>8:
            raise ValueError('TCRR_TRAIN_WORKERS must be 0..8')
    if 'TCRR_VAL_WORKERS' in os.environ:
        val=int(os.environ['TCRR_VAL_WORKERS'])
        if not (0<=val<=8): raise ValueError('TCRR_VAL_WORKERS must be 0..8')
        cfg['runtime']['val_num_workers']=val
    if 'TCRR_PERSISTENT_WORKERS' in os.environ:
        val=int(os.environ['TCRR_PERSISTENT_WORKERS'])
        if val not in (0,1): raise ValueError('TCRR_PERSISTENT_WORKERS must be 0 or 1')
        cfg['runtime']['persistent_workers']=bool(val)
    if 'TCRR_PREFETCH_FACTOR' in os.environ:
        val=int(os.environ['TCRR_PREFETCH_FACTOR'])
        if not (1<=val<=8): raise ValueError('TCRR_PREFETCH_FACTOR must be 1..8')
        cfg['runtime']['prefetch_factor']=val
    print('V5_LOADER_CONFIG '+json.dumps({
        'batch_size':cfg['training']['batch_size'],
        'num_workers':cfg['runtime'].get('num_workers',0),
        'val_num_workers':cfg['runtime'].get('val_num_workers',0),
        'persistent_workers':cfg['runtime'].get('persistent_workers',False),
        'prefetch_factor':cfg['runtime'].get('prefetch_factor',2),
        'pin_memory':cfg['runtime'].get('pin_memory',True),
    }),flush=True)
    for key,value in (('diagnostic_negative_max_batches',a.diagnostic_max_batches),
                      ('diagnostic_primary_max_batches',a.diagnostic_primary_max_batches)):
        if value is not None:
            if value < 0: p.error('Diagnostic batch limits must be >= 0')
            cfg['v5'][key]=value
    ck=resolve_stla_checkpoint(cfg,a.backbone_checkpoint)
    os.environ['BACKBONE_CHECKPOINT']=str(ck)
    seed_all(a.seed)
    stamp=a.run_stamp or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run=(Path(a.resume).resolve().parent if a.resume else
         Path(cfg['project']['exp_root'])/a.dataset/'v5'/('smoke' if a.smoke else 'formal')/f'{stamp}_seed{a.seed}_full')
    if not a.resume and run.exists():raise FileExistsError(f'Run already exists, preventing overwrite: {run}')
    run.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache=cache_path(cfg,'formal')
    sha=file_sha256(ck)[:16]
    adapter=Path(cfg['project']['retrieval_root'])/a.dataset/'v5'/('shared_adapters_smoke' if a.smoke else 'shared_adapters')/f'{sha}_seed{a.seed}'/'adapter.pt'
    if a.resume:
        portable=run/'s0_shared_adapter.pt'
        if not portable.is_file():raise FileNotFoundError(portable)
    else:
        fit_shared_adapter(cfg,ck,cache,adapter,device,smoke=a.smoke)
        portable=run/'s0_shared_adapter.pt';shutil.copy2(adapter,portable)
    cfg['runtime']['teacher_adapter_checkpoint']=str(portable)
    cfg['runtime']['teacher_fingerprint']=file_sha256(portable)
    rt=ensure_cache(cfg,cache,'train',smoke=a.smoke)
    rv=ensure_cache(cfg,cache,'val',smoke=a.smoke)
    checkpoint=train_v5(cfg,cache,ck,rt,rv,run,portable,smoke=a.smoke,resume=a.resume)
    print('V5_BEST_CHECKPOINT',checkpoint,flush=True)
    # Training and diagnostics share the same PBS allocation. Run on validation
    # before evaluating the formal test sets; subprocess errors fail the job.
    if not a.smoke and not a.skip_diagnostics and cfg['v5'].get('auto_diagnostics',True):
        diag_args=[sys.executable,'-u',str(ROOT/'scripts/auto_diagnose_v5.py'),
                   '--run-dir',str(run),
                   '--batch-size',str(cfg['v5'].get('diagnostic_batch_size',8)),
                   '--negative-max-batches',str(cfg['v5'].get('diagnostic_negative_max_batches',64)),
                   '--primary-max-batches',str(cfg['v5'].get('diagnostic_primary_max_batches',0))]
        print('V5_AUTO_DIAGNOSTICS_START run='+str(run),flush=True)
        subprocess.check_call(diag_args,cwd=ROOT)
        print('V5_AUTO_DIAGNOSTICS_PASS run='+str(run),flush=True)
    if not a.smoke and not a.skip_evaluate:
        c12=run/'best_c12.pt'
        if c12.is_file():
            args=[sys.executable,'-u',str(ROOT/'scripts/evaluate_v5.py'),
                  '--checkpoint',str(c12),'--controller-mode','fixed','--diagnostics']
            subprocess.check_call(args,cwd=ROOT)
        for stage in ('controller','joint'):
            candidate=run/f'best_{stage}.pt'
            if candidate.is_file():
                args=[sys.executable,'-u',str(ROOT/'scripts/evaluate_v5.py'),
                      '--checkpoint',str(candidate),'--modes',*list(THREE_SWITCH_MODES),
                      '--controller-mode','auto','--diagnostics']
                subprocess.check_call(args,cwd=ROOT)
        print('V5_AUTO_EVALUATE_PASS',flush=True)
if __name__=='__main__':main()
