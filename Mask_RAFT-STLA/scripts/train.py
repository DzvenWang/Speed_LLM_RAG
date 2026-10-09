#!/usr/bin/env python3
from pathlib import Path
import argparse,datetime,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mask_raft_stla.config import load_config,cache_path,resolve_stla_checkpoint
from mask_raft_stla.trainer import train
from scripts.build_retrieval_cache import ensure_cache

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--target',choices=['speed','flow']);p.add_argument('--seed',type=int,default=10)
    p.add_argument('--dataset');p.add_argument('--backbone-checkpoint');p.add_argument('--config');p.add_argument('--smoke',action='store_true')
    p.add_argument('--run-stamp');p.add_argument('--batch-size',type=int);p.add_argument('--num-workers',type=int)
    p.add_argument('--ablation',choices=['full','no_retrieval','no_evidence','no_mask_replay','no_rerank','standard_future','shuffled'])
    p.add_argument('--resume',help='Path to last.pt from a prior v2.1 run')
    p.add_argument('--skip-evaluate',action='store_true')
    a=p.parse_args()
    if a.target:os.environ['TRAFFIDENT_TARGET']=a.target
    cfg=load_config(a.config)
    if a.dataset:cfg['experiment']['dataset']=a.dataset
    cfg['experiment']['seed']=a.seed
    if a.batch_size:cfg['training']['batch_size']=a.batch_size
    if a.num_workers is not None:cfg['runtime']['num_workers']=a.num_workers
    if a.ablation:cfg['training']['ablation']=a.ablation
    cache=cache_path(cfg,'formal');backbone=resolve_stla_checkpoint(cfg,a.backbone_checkpoint)
    rt=ensure_cache(cfg,cache,'train',smoke=a.smoke);rv=ensure_cache(cfg,cache,'val',smoke=a.smoke)
    stage='formal_smoke' if a.smoke else 'formal';stamp=a.run_stamp or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run=Path(a.resume).resolve().parent if a.resume else Path(cfg['project']['exp_root'])/cfg['experiment']['dataset']/stage/f'{stamp}_seed{a.seed}_{cfg["training"]["ablation"]}'
    best=train(cfg,cache,backbone,rt,rv,run,a.smoke,resume=a.resume);print(f'BEST_CHECKPOINT={best}',flush=True)
    if not a.smoke and not a.skip_evaluate:
        cmd=[sys.executable,'-u',str(ROOT/'scripts/evaluate.py'),'--checkpoint',str(best),
             '--severities',*[str(s) for s in cfg['experiment']['severities']]]
        subprocess.check_call(cmd,cwd=ROOT)
        print('AUTO_EVALUATE_PASS',best,flush=True)
    else:print('AUTO_EVALUATE_SKIP reason=smoke_or_cli',flush=True)
if __name__=='__main__':main()
