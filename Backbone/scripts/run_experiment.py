#!/usr/bin/env python3
from pathlib import Path
import argparse, datetime, os, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT.parent/'TraffiDent_SupportBench'))
from backbone.utils import load_yaml, install_tee, log
from backbone.shared_cache import cache_path, validate_shared_cache
from backbone.runner import train_patchstg
from supportbench.dataset import SupportBench

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['development','formal'],required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--config',default=str(ROOT/'configs/backbone.yaml')); ap.add_argument('--run-stamp'); a=ap.parse_args(); cfg=load_yaml(a.config)
    if a.stage=='development' and a.seed!=int(cfg['experiment']['development_seed']): raise SystemExit(f'development seed fixed to {cfg["experiment"]["development_seed"]}')
    if a.stage=='formal' and a.seed not in [int(x) for x in cfg['experiment']['formal_seeds']]: raise SystemExit('formal seed not configured')
    stamp=a.run_stamp or datetime.datetime.now().strftime('%Y%m%d_%H%M%S'); log_path=Path(cfg['project']['log_root'])/f'PatchSTG.{a.stage}.seed{a.seed}.{stamp}.log'
    if os.environ.get('BACKBONE_PBS_LIVE_LOG','').strip() not in {'1','true','TRUE','yes','YES'}:
        install_tee(log_path); print('LOG_FILE',log_path)
    else:
        print('LOG_MODE PBS_LIVE_ONLY')
    print('BACKBONE_VERSION',cfg['version'])
    subprocess.check_call([sys.executable,str(ROOT/'scripts/source_lock.py'),'verify','--config',a.config]); validate_shared_cache(cfg,a.stage); cache=cache_path(cfg,a.stage)
    b=SupportBench(cfg['experiment']['subgraph'],root=cfg['project']['benchmark_root']); st=b.stats['channels'][b.manifest['primary_channel']]
    run=Path(cfg['project']['exp_root'])/'PatchSTG'/cfg['experiment']['subgraph']/a.stage/f'{stamp}_seed{a.seed}'; log(f'run_dir={run}'); log(f'cache_dir={cache}')
    train_patchstg(cfg,cache,run,ROOT/'vendor'/'PatchSTG',b.adj,b.nodes,float(st['mean']),float(st['std']),a.seed,a.stage)
    checkpoint = run / 'best.pt'
    if not checkpoint.exists():
        raise RuntimeError(f'Training completed but best checkpoint is missing: {checkpoint}')
    log(f'BEST_CHECKPOINT={checkpoint}')
    log(f'TRAINING_PASS stage={a.stage} seed={a.seed}')
if __name__=='__main__': main()
