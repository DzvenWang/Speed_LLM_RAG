#!/usr/bin/env python3
"""Chronological STLAformer OOF: plan, train one independently scheduled fold, or finalize.

Each fold job owns a distinct folder; finalization runs ONLY after every producer
passes strict checkpoint/provenance checks. A shared fold plan is immutable during
parallel jobs. Never train folds sequentially when --fold-id is provided.
"""
import _bootstrap
import argparse, json, torch
from pathlib import Path
from fare.config import read_config,cache_dir,backbone_checkpoint,root,hash_file
from fare.oof import (save_plan, train_fold, export_paired,verify_paired,sha_json,
                      validate_completed_fold, load_frozen_plan)
from fare.protocol import SupportDataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='FARE/configs/formal.yaml')
    p.add_argument('--dataset',default='TraffiDent_Net_06')
    p.add_argument('--target',choices=('speed','flow','both'),default='both')
    p.add_argument('--nfolds',type=int,default=4)
    p.add_argument('--warmup',type=float,default=.20)
    p.add_argument('--purge',type=int,default=12)
    p.add_argument('--seed',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--epochs',type=int)
    p.add_argument('--output-dir',default='FARE_cache/oof_v503')
    p.add_argument('--speed-checkpoint')
    p.add_argument('--flow-checkpoint')
    actions=p.add_mutually_exclusive_group()
    actions.add_argument('--plan-only',action='store_true')
    actions.add_argument('--fold-id',type=int,help='Train exactly one fold; do NOT export or build memory')
    actions.add_argument('--finalize-only',action='store_true',help='Verify all completed folds, export paired OOF and build memory')
    p.add_argument('--skip-memory',action='store_true')
    a=p.parse_args()
    if a.nfolds<1 or a.purge<0 or a.batch_size<1 or a.workers<0 or not (0<a.warmup<1) or (a.epochs is not None and a.epochs<1):
        p.error('Invalid fold settings')
    if a.fold_id is not None and (a.target=='both' or a.fold_id<0 or a.fold_id>=a.nfolds):
        p.error('--fold-id requires one target and an ID in 0..nfolds-1')
    output=Path(a.output_dir)
    if not output.is_absolute():output=root()/output
    targets=('speed','flow') if a.target=='both' else (a.target,)
    from fare.oof import _external_backbone
    _,_,_,recipe_loader,_=_external_backbone()
    expected_recipe=dict(recipe_loader(root()/'Backbone/configs/model_recipes.yaml')['models']['STLAformer'])
    if a.epochs is not None:expected_recipe['max_epochs']=a.epochs
    expected_recipe['batch_size']=a.batch_size
    recipe_sha=sha_json(expected_recipe)
    for target in targets:
        cfg=read_config(a.config,a.dataset,target,seed=a.seed)
        cache=cache_dir(cfg,'formal')
        ref=backbone_checkpoint(cfg,getattr(a,f'{target}_checkpoint'))
        ck=torch.load(ref,map_location='cpu',weights_only=False)
        if ck.get('model')!='STLAformer' or ck.get('target')!=target:
            raise RuntimeError('Reference STLAformer checkpoint target mismatch')
        planpath=output/target/a.dataset/'fold_plan.json'
        if a.plan_only:
            # Only this phase is allowed to write a fold plan (single process).
            # Reject a change of plan while a prior run still exists.
            if planpath.exists():
                old=json.loads(planpath.read_text())
                candidate=save_plan(cache,a.config,target,a.dataset,planpath.with_name('fold_plan.candidate.json'),a.nfolds,a.warmup,a.purge)
                planpath.with_name('fold_plan.candidate.json').unlink()
                if old['plan_sha256']!=candidate['plan_sha256']:
                    raise RuntimeError(f'Existing fold plan differs; choose a new --output-dir or archive {planpath}')
                plan=old
            else:
                plan=save_plan(cache,a.config,target,a.dataset,planpath,a.nfolds,a.warmup,a.purge)
        else:
            plan=load_frozen_plan(planpath,cache,target,a.dataset,a.nfolds,a.warmup,a.purge)
        print('FARE_OOF_PLAN '+json.dumps({'target':target,'folds':a.nfolds,
              'total_oof':sum(f['score_count'] for f in plan['folds']),
              'first_scored_window':plan['folds'][0]['evaluate_from'],
              'train_rows':plan['train_rows'],'folds_summary':[{'id':f['id'],'fit':f['fit_count'],'val':f['val_count'],'score':f['score_count'],'start':f['evaluate_from'],'end':f['evaluate_until']} for f in plan['folds']],'reference_checkpoint':str(ref),
              'reference_hash':hash_file(ref),'plan_sha256':plan['plan_sha256'],
              'plan_path':str(planpath)}),flush=True)
        if a.plan_only:continue
        def ckpath(fold):return output/target/a.dataset/f'fold_{fold["id"]:02d}'/'best.pt'
        if a.fold_id is not None:
            fold=plan['folds'][a.fold_id]
            checkpoint=ckpath(fold)
            if validate_completed_fold(checkpoint,plan,fold,ref,recipe_sha):
                print(f'FARE_OOF_ALREADY_COMPLETED target={target} fold={a.fold_id}',flush=True)
            else:
                # If stale/incompatible artifacts exist, fail rather than silently overwriting.
                if checkpoint.with_name('COMPLETED.json').exists():
                    raise RuntimeError(f'Fold completed marker invalid: {checkpoint.parent}. Move stale artifacts first.')
                train_fold(cfg,plan,fold,ref,checkpoint.parent,seed=a.seed,
                           epochs=a.epochs,batch_size=a.batch_size,workers=a.workers)
                if not validate_completed_fold(checkpoint,plan,fold,ref,recipe_sha):
                    raise RuntimeError(f'Fold did not pass completion validation: {checkpoint}')
            print(f'FARE_OOF_FOLD_JOB_PASS target={target} fold={a.fold_id}',flush=True)
            continue
        models=[]
        for fold in plan['folds']:
            path=ckpath(fold)
            if a.finalize_only:
                if not validate_completed_fold(path,plan,fold,ref,recipe_sha):
                    raise RuntimeError(f'Cannot finalize before verified completion: {path}')
            elif not validate_completed_fold(path,plan,fold,ref,recipe_sha):
                train_fold(cfg,plan,fold,ref,path.parent,seed=a.seed,
                           epochs=a.epochs,batch_size=a.batch_size,workers=a.workers)
            models.append(path)
        paired=root()/'FARE_cache'/f'oof_{target}_paired.npz'
        export_paired(plan,models,paired,batch_size=a.batch_size)
        verify_paired(paired,SupportDataset(cache,'train'))
        if not a.skip_memory:
            from fare.pipeline import prepare
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            _,memory,_,_,_,meta=prepare(cfg,device,str(ref),oos_forecast_file=str(paired))
            print('FARE_OOF_MEMORY_PASS '+json.dumps({'target':target,'rows':memory.m,
                  'memory_dir':meta['memory_path'],
                  'residual_provenance':meta['residual_provenance']}),flush=True)
        if torch.cuda.is_available():torch.cuda.empty_cache()
    print('FARE_V503_OOF_PIPELINE_COMPLETE',flush=True)

if __name__=='__main__':main()
