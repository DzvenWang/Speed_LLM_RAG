#!/usr/bin/env python3
from pathlib import Path
import argparse, datetime, os, sys
ROOT=Path(__file__).resolve().parents[1]; REPO=ROOT.parent
sys.path[:0]=[str(ROOT),str(REPO/'Base_SupportBench')]
from backbone.model_runner import MODELS as GENERIC_MODELS, train_model, smoke_model
from backbone.runner import train_patchstg, smoke_patchstg
from backbone.shared_cache import cache_path, validate_shared_cache
from backbone.utils import load_yaml, log
from supportbench.dataset import SupportBench

MODELS = (*GENERIC_MODELS[:4], 'PatchSTG', *GENERIC_MODELS[4:])

def main():
    ap=argparse.ArgumentParser(description='User-selected single-model Formal runner; no auto-selection and no freeze.')
    ap.add_argument('--model',required=True,choices=MODELS); ap.add_argument('--target',choices=['speed','flow']); ap.add_argument('--seed',type=int,default=10); ap.add_argument('--run-stamp')
    ap.add_argument('--config',default=str(ROOT/'configs/backbone.yaml')); ap.add_argument('--recipes',default=str(ROOT/'configs/model_recipes.yaml'))
    ap.add_argument('--smoke',action='store_true',help='Run bounded real-Formal smoke; does not auto-evaluate or freeze')
    ap.add_argument('--smoke-train-batches',type=int,default=2); ap.add_argument('--smoke-val-batches',type=int,default=1)
    a=ap.parse_args()
    if a.target: os.environ['TRAFFIDENT_TARGET']=a.target
    cfg=load_yaml(a.config); recipes=load_yaml(a.recipes)
    if a.seed != 10: raise SystemExit('Formal seed is fixed to 10')
    target=str(cfg['experiment']['target'])
    if target not in ('speed','flow'): raise SystemExit(f'TRAFFIDENT_TARGET must resolve to speed or flow, got {target!r}')
    if a.target and target != a.target: raise SystemExit(f'target resolution mismatch: CLI={a.target} config={target}')
    validate_shared_cache(cfg,'formal'); cache=cache_path(cfg,'formal'); bench=SupportBench(cfg['experiment']['subgraph'],root=cfg['project']['benchmark_root'])
    st=bench.stats['channels'][bench.manifest['primary_channel']]; stamp=a.run_stamp or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    stage_dir='formal_smoke' if a.smoke else 'formal'
    run=Path(cfg['project']['exp_root'])/a.model/cfg['experiment']['subgraph']/stage_dir/f'{stamp}_seed{a.seed}'
    log(f'USER_SELECTED_FORMAL model={a.model} target={target} seed={a.seed} smoke={a.smoke} run_dir={run} cache_dir={cache}')
    if a.smoke:
        if a.model == 'PatchSTG':
            smoke_patchstg(cfg,cache,run,ROOT/'vendor'/'PatchSTG',bench.adj,bench.nodes,float(st['mean']),float(st['std']),a.seed,'formal',a.smoke_train_batches,a.smoke_val_batches)
        else:
            smoke_model(cfg,recipes,a.model,cache,run,ROOT/'vendor',bench,float(st['mean']),float(st['std']),a.seed,a.smoke_train_batches,a.smoke_val_batches)
        print(f'TRAINING_SMOKE_PASS model={a.model} target={target} stage=formal seed={a.seed}',flush=True)
        return
    if a.model == 'PatchSTG':
        train_patchstg(cfg,cache,run,ROOT/'vendor'/'PatchSTG',bench.adj,bench.nodes,float(st['mean']),float(st['std']),a.seed,'formal')
    else:
        train_model(cfg,recipes,a.model,cache,run,ROOT/'vendor',bench,float(st['mean']),float(st['std']),a.seed)
    ck=run/'best.pt'
    if not ck.exists(): raise RuntimeError(f'best.pt missing after Formal training: {ck}')
    print(f'BEST_CHECKPOINT={ck}',flush=True); print(f'FORMAL_TRAINING_PASS model={a.model} target={target} seed={a.seed}',flush=True)
if __name__=='__main__': main()
