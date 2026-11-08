#!/usr/bin/env python3
"""TCRR v4 formal training: S0 -> cache -> S2/S3/S4 -> evaluation."""
from __future__ import annotations
import argparse,datetime,os,sys,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tcrr_stla.config import load_config,cache_path,resolve_stla_checkpoint,file_sha256
from tcrr_stla.v4.trainer import fit_shared_adapter,train_v4
from scripts.build_retrieval_cache import ensure_cache
import torch

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--target',choices=('speed','flow'),required=True)
    p.add_argument('--dataset',default='TraffiDent_Net_06')
    p.add_argument('--seed',type=int,default=10)
    p.add_argument('--backbone-checkpoint',required=True)
    p.add_argument('--shared-adapter-checkpoint')
    p.add_argument('--config')
    p.add_argument('--batch-size',type=int)
    p.add_argument('--run-stamp')
    p.add_argument('--resume')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--skip-evaluate',action='store_true')
    p.add_argument('--ablation',default='full',choices=('full','no_failure_residual',
        'no_normal_residual','no_support_conditioning','no_utility_ranking','no_pair_interaction',
        'shuffled_cf_pairing','no_gain_harm','no_risk','similarity_only'))
    a=p.parse_args()
    os.environ['TRAFFIDENT_TARGET']=a.target
    cfg=load_config(a.config or ROOT/'configs'/'tcrr_stla_v4.yaml')
    cfg['experiment'].update(dataset=a.dataset,target=a.target,seed=a.seed)
    if a.batch_size:cfg['training']['batch_size']=a.batch_size
    cfg['training']['ablation']=a.ablation
    backbone=resolve_stla_checkpoint(cfg,a.backbone_checkpoint)
    os.environ['BACKBONE_CHECKPOINT']=str(backbone)
    cache=cache_path(cfg,'formal')
    stamp=a.run_stamp or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run=Path(a.resume).resolve().parent if a.resume else (Path(cfg['project']['exp_root'])/a.dataset/'v4'/
             ('formal_smoke' if a.smoke else 'formal')/f'{stamp}_seed{a.seed}_{a.ablation}')
    run.mkdir(parents=True,exist_ok=True)
    from tcrr_stla.v4.trainer import seed_all
    seed_all(a.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sh=file_sha256(backbone)[:16]
    shared=Path(cfg['project']['retrieval_root'])/a.dataset/'v4'/'shared_adapters'/f'{sh}_seed{a.seed}'/'adapter.pt'
    if a.shared_adapter_checkpoint:
        shared=Path(a.shared_adapter_checkpoint).resolve()
        if not shared.is_file():raise FileNotFoundError(shared)
    else:
        fit_shared_adapter(cfg,backbone,cache,shared,device,smoke=a.smoke)
    portable=run/'s0_shared_adapter.pt'
    if shared.resolve()!=portable.resolve():shutil.copy2(shared,portable)
    cfg['runtime']['teacher_adapter_checkpoint']=str(portable)
    cfg['runtime']['teacher_fingerprint']=file_sha256(portable)
    # Adapted S0 is now fixed; shortlist/forecast-pool fingerprints include it.
    retrieval_train=ensure_cache(cfg,cache,'train',smoke=a.smoke)
    retrieval_val=ensure_cache(cfg,cache,'val',smoke=a.smoke)
    best=train_v4(cfg,cache,backbone,retrieval_train,retrieval_val,run,portable,a.smoke,a.resume)
    print('V4_BEST_CHECKPOINT',best,flush=True)
    if not a.smoke and not a.skip_evaluate:
        cmd=[sys.executable,'-u',str(ROOT/'scripts/evaluate_v4.py'),'--checkpoint',str(best),
             '--batch-size',str(cfg['runtime'].get('eval_batch_size',8)),'--severities',
             *[str(v) for v in cfg['experiment']['severities']]]
        subprocess.check_call(cmd,cwd=ROOT)
        print('V4_AUTO_EVALUATE_PASS',flush=True)

if __name__=='__main__':main()
