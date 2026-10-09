#!/usr/bin/env python3
from pathlib import Path
import argparse, datetime, os, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]; REPO=ROOT.parent
sys.path[:0]=[str(ROOT),str(REPO/'Base_SupportBench')]
from backbone.model_runner import MODELS as GENERIC_MODELS, train_model, smoke_model
from backbone.runner import train_patchstg, smoke_patchstg
from backbone.shared_cache import cache_path, validate_shared_cache
from backbone.utils import load_yaml, log
from backbone.path_pointers import portable_project_path
from backbone.training_modes import require_mode, run_stage_directory
from supportbench.dataset import SupportBench

MODELS = (*GENERIC_MODELS[:4], 'PatchSTG', *GENERIC_MODELS[4:])

class _Tee:
    def __init__(self, *streams): self.streams=streams
    def write(self, data):
        for stream in self.streams:
            stream.write(data); stream.flush()
        return len(data)
    def flush(self):
        for stream in self.streams: stream.flush()
    def isatty(self): return any(getattr(s, 'isatty', lambda: False)() for s in self.streams)


def _enable_local_live_log(cfg, model, target, seed, stamp, smoke, mode):
    # PBS already timestamps and tees the complete job stream, including automatic evaluation.
    # Only create this Python-side tee for direct/local invocations to avoid duplicate Gadi logs.
    if os.environ.get('BACKBONE_PBS_LIVE_LOG') == '1':
        return None, None
    log_dir=Path(cfg['project']['log_root'])/model
    log_dir.mkdir(parents=True,exist_ok=True)
    stage_tag=run_stage_directory('formal',mode,smoke)
    path=log_dir/f'{model}.{target}.{stage_tag}.seed{seed}.local.{stamp}.live.log'
    fh=path.open('a',encoding='utf-8',buffering=1)
    sys.stdout=_Tee(sys.__stdout__,fh); sys.stderr=_Tee(sys.__stderr__,fh)
    print(f'LOCAL_LIVE_LOG={path}',flush=True)
    return path, fh


def main():
    ap=argparse.ArgumentParser(description='User-selected single-model Formal runner; Formal training auto-evaluates s20/s40/s60/s90 and publishes a latest-checkpoint pointer.')
    ap.add_argument('--model',required=True,choices=MODELS); ap.add_argument('--target',choices=['speed','flow']); ap.add_argument('--seed',type=int,default=10); ap.add_argument('--run-stamp')
    ap.add_argument('--config',default=str(ROOT/'configs/backbone.yaml')); ap.add_argument('--recipes',default=str(ROOT/'configs/model_recipes.yaml'))
    ap.add_argument('--train-data-mode',choices=('incomplete','complete'),default=None,help='incomplete (existing s40 training) or complete (remove artificial history masks); default incomplete')
    ap.add_argument('--smoke',action='store_true',help='Run bounded real-Formal training smoke; smoke controller handles bounded evaluation')
    ap.add_argument('--smoke-train-batches',type=int,default=2); ap.add_argument('--smoke-val-batches',type=int,default=1)
    a=ap.parse_args()
    config_path=Path(a.config).expanduser()
    if not config_path.is_absolute():
        cwd_candidate=(Path.cwd()/config_path).resolve()
        root_candidate=(ROOT/config_path).resolve()
        config_path=cwd_candidate if cwd_candidate.exists() else root_candidate
    else: config_path=config_path.resolve()
    recipes_path=Path(a.recipes).expanduser()
    if not recipes_path.is_absolute():
        cwd_candidate=(Path.cwd()/recipes_path).resolve()
        root_candidate=(ROOT/recipes_path).resolve()
        recipes_path=cwd_candidate if cwd_candidate.exists() else root_candidate
    else: recipes_path=recipes_path.resolve()
    if not config_path.is_file(): raise SystemExit(f'Config not found: {config_path}')
    if not recipes_path.is_file(): raise SystemExit(f'Recipes not found: {recipes_path}')
    if a.target: os.environ['TRAFFIDENT_TARGET']=a.target
    cfg=load_yaml(config_path); recipes=load_yaml(recipes_path)
    train_data_mode=require_mode(a.train_data_mode or cfg.get('experiment',{}).get('train_data_mode','incomplete'))
    if a.seed != 10: raise SystemExit('Formal seed is fixed to 10')
    target=str(cfg['experiment']['target'])
    if target not in ('speed','flow'): raise SystemExit(f'TRAFFIDENT_TARGET must resolve to speed or flow, got {target!r}')
    if a.target and target != a.target: raise SystemExit(f'target resolution mismatch: CLI={a.target} config={target}')
    validate_shared_cache(cfg,'formal'); cache=cache_path(cfg,'formal'); bench=SupportBench(cfg['experiment']['subgraph'],root=cfg['project']['benchmark_root'])
    st=bench.stats['channels'][bench.manifest['primary_channel']]; stamp=a.run_stamp or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    _live_path, _live_fh = _enable_local_live_log(cfg,a.model,target,a.seed,stamp,a.smoke,train_data_mode)
    stage_dir=run_stage_directory('formal',train_data_mode,a.smoke)
    run=Path(cfg['project']['exp_root'])/a.model/cfg['experiment']['subgraph']/stage_dir/f'{stamp}_seed{a.seed}'
    log(f'USER_SELECTED_FORMAL model={a.model} target={target} seed={a.seed} train_data_mode={train_data_mode} smoke={a.smoke} run_dir={run} cache_dir={cache}')
    if a.smoke:
        if a.model == 'PatchSTG':
            smoke_patchstg(cfg,cache,run,ROOT/'vendor'/'PatchSTG',bench.adj,bench.nodes,float(st['mean']),float(st['std']),a.seed,'formal',a.smoke_train_batches,a.smoke_val_batches,train_data_mode=train_data_mode)
        else:
            smoke_model(cfg,recipes,a.model,cache,run,ROOT/'vendor',bench,float(st['mean']),float(st['std']),a.seed,a.smoke_train_batches,a.smoke_val_batches,train_data_mode=train_data_mode)
        print(f'TRAINING_SMOKE_PASS model={a.model} target={target} stage=formal seed={a.seed}',flush=True)
        return
    if a.model == 'PatchSTG':
        train_patchstg(cfg,cache,run,ROOT/'vendor'/'PatchSTG',bench.adj,bench.nodes,float(st['mean']),float(st['std']),a.seed,'formal',train_data_mode=train_data_mode)
    else:
        train_model(cfg,recipes,a.model,cache,run,ROOT/'vendor',bench,float(st['mean']),float(st['std']),a.seed,train_data_mode=train_data_mode)
    ck=run/'best.pt'
    if not ck.exists(): raise RuntimeError(f'best.pt missing after Formal training: {ck}')
    print(f'BEST_CHECKPOINT={ck}',flush=True); print(f'FORMAL_TRAINING_PASS model={a.model} target={target} seed={a.seed}',flush=True)
    print(f'AUTO_EVAL_START model={a.model} target={target} checkpoint={ck}',flush=True)
    cmd=[sys.executable,'-u',str(ROOT/'scripts/post_training_evaluate.py'),'--model',a.model,'--checkpoint',str(ck.resolve()),'--seed',str(a.seed),'--run-stamp',stamp,'--config',str(config_path)]
    subprocess.check_call(cmd,cwd=ROOT)
    print(f'AUTO_EVAL_PASS model={a.model} target={target} severities=20,40,60,90',flush=True)
    formal_root=Path(cfg['project']['exp_root'])/a.model/cfg['experiment']['subgraph']/stage_dir
    formal_root.mkdir(parents=True,exist_ok=True)
    project_root=Path(cfg['project']['root']).expanduser().resolve()
    run_ptr=portable_project_path(run,project_root)
    ck_ptr=portable_project_path(ck,project_root)
    (formal_root/'LATEST_RUN.txt').write_text(run_ptr+'\n',encoding='utf-8')
    (formal_root/'LATEST_CHECKPOINT.txt').write_text(ck_ptr+'\n',encoding='utf-8')
    print(f'LATEST_BACKBONE_POINTER model={a.model} run={run_ptr} checkpoint={ck_ptr}',flush=True)
if __name__=='__main__': main()
