#!/usr/bin/env python3
"""Evaluate a true, nested hidden-cell restoration curve."""
from pathlib import Path
import argparse,sys,time
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from oracle.backbone import load_patchstg_adapter
from oracle.config import load_config
from oracle.contract import assert_observation_experiment_ready,cache_dir
from oracle.data import OracleCachedDataset
from oracle.deep import make_inputs
from oracle.metrics import MetricAccumulator,improvement_pct
from oracle.observation import apply_true_coverage,build_observation_aware
from oracle.utils import deterministic_split,log,seed_all,sha256_file,write_json


def part_flags(indices,dev_set,final_set):
    ids=np.asarray(indices,np.int64)
    return {'all':np.ones(len(ids),bool),
            'dev':np.asarray([int(x) in dev_set for x in ids],bool),
            'final':np.asarray([int(x) in final_set for x in ids],bool)}


def parse_models(items):
    out=[]
    for item in items:
        if '=' not in item: raise ValueError(f'--model must be NAME=CHECKPOINT, got {item}')
        name,path=item.split('=',1)
        if not name or not path: raise ValueError(f'Invalid --model {item}')
        if any(name==old for old,_ in out) or name=='reference': raise ValueError(f'Duplicate/reserved model name {name}')
        out.append((name,Path(path)))
    return out


def load_models(adapter,base_path,items,device,history,cache_identity):
    base_sha=sha256_file(base_path); models=[('reference',adapter.model,False,None)]
    for name,path in items:
        if not path.exists(): raise FileNotFoundError(path)
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck.get('format')!='patchstg_observation_aware_v1': raise RuntimeError(f'Unsupported checkpoint format: {path}')
        if str(ck.get('base_checkpoint_sha256'))!=base_sha:
            raise RuntimeError(f'{path} was not trained from the selected reference PatchSTG')
        trained_ident=ck.get('cache_identity',{})
        if str(trained_ident.get('cache_fingerprint'))!=str(cache_identity.get('cache_fingerprint')):
            raise RuntimeError(f'{path} cache fingerprint does not match this coverage run')
        model=build_observation_aware(adapter.model,int(ck.get('history',history)),int(ck['mask_hidden']))
        model.load_state_dict(ck['state_dict']); model.to(device).eval()
        models.append((name,model,True,{'checkpoint':str(path),'checkpoint_sha256':sha256_file(path),'variant':ck.get('variant'),'best_epoch':ck.get('best_epoch')}))
    return models


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',default='development',choices=['development','formal'])
    ap.add_argument('--run-tag',required=True)
    ap.add_argument('--checkpoint',default=None,help='Reference PatchSTG checkpoint override')
    ap.add_argument('--model',action='append',default=[],help='Learned model as NAME=best.pt; repeatable')
    a=ap.parse_args()

    cfg=load_config(ROOT); seed=int(cfg['experiment']['seed']); seed_all(seed)
    ident=assert_observation_experiment_ready(cfg,a.stage); cache=cache_dir(cfg,a.stage)
    out=Path(cfg['project']['exp_root'])/cfg['experiment']['subgraph']/a.stage/a.run_tag/'coverage_curve'
    out.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    adapter,base_path,base_ck,source=load_patchstg_adapter(cfg,a.stage,device,a.checkpoint)
    mean,std=float(base_ck['mean']),float(base_ck['std'])
    ds=OracleCachedDataset(cache,'val'); rt=cfg['runtime']
    kw=dict(dataset=ds,batch_size=int(rt['batch_size']),shuffle=False,drop_last=False,
            num_workers=int(rt['num_workers']),pin_memory=torch.cuda.is_available())
    if int(rt['num_workers'])>0: kw.update(persistent_workers=True,prefetch_factor=int(rt['prefetch_factor']))
    dl=DataLoader(**kw)
    dev_ids,final_ids=deterministic_split(len(ds),cfg['experiment']['dev_queries'],cfg['experiment']['final_val_queries'],seed)
    dev_set,final_set=set(dev_ids.tolist()),set(final_ids.tolist())
    fractions=[float(x) for x in cfg['coverage_curve']['fractions']]
    if fractions!=sorted(set(fractions)) or not fractions or fractions[0]!=0.0 or fractions[-1]!=1.0:
        raise RuntimeError('coverage_curve.fractions must be unique, sorted, and span 0 to 1')
    restore_seed=int(cfg['coverage_curve']['seed'])
    models=load_models(adapter,base_path,parse_models(a.model),device,cfg['experiment']['history'],ident)
    results={}; t0=time.time()

    for model_name,model,aware,model_meta in models:
        model.eval(); metrics={f:{p:MetricAccumulator(cfg['experiment']['horizon']) for p in ['all','dev','final']} for f in fractions}
        selected_counts={f:0 for f in fractions}; restorable_count=0
        for bi,b in enumerate(dl,1):
            idx=b['index']; flags=part_flags(idx.numpy(),dev_set,final_set)
            x0=b['x_obs'].to(device,non_blocking=True).float()
            xc=b['x_complete'].to(device,non_blocking=True).float()
            observed=b['history_observed'].to(device,non_blocking=True).bool()
            restorable=b['restorable'].to(device,non_blocking=True).bool()
            restorable_count+=int(restorable.sum().item())
            y=b['y_raw'].numpy(); valid=b['future_valid'].numpy(); direct=b['direct_mask'].numpy(); support=b['support_mask'].numpy()
            for frac in fractions:
                xr,mr,selected=apply_true_coverage(x0,xc,observed,restorable,idx,frac,restore_seed)
                selected_counts[frac]+=int(selected.sum().item())
                x,te=make_inputs(xr,b['tod_hist'],b['dow_hist'],device)
                with torch.no_grad():
                    pn=model(x,te,mr) if aware else model(x,te)
                    if pn.ndim==4 and pn.shape[-1]==1: pn=pn.squeeze(-1)
                    pred=(pn*std+mean).detach().cpu().numpy()
                for part,row_mask in flags.items(): metrics[frac][part].update(pred,y,valid,direct,support,row_mask)
            if bi%20==0 or bi==len(dl): log(f'COVERAGE {model_name} batch={bi}/{len(dl)} elapsed={time.time()-t0:.1f}s')
        summaries={str(f):{p:metrics[f][p].summary() for p in ['all','dev','final']} for f in fractions}
        base=summaries[str(fractions[0])]
        gains={str(f):{p:{k:improvement_pct(base[p][k],summaries[str(f)][p][k]) for k in
                  ['overall_mae','masked_mae','support_mae','overall_rmse','masked_rmse','support_rmse']}
                  for p in ['all','dev','final']} for f in fractions}
        realized={str(f):{'selected_cells':selected_counts[f],'restorable_cells':restorable_count,
                          'realized_fraction':selected_counts[f]/max(restorable_count,1)} for f in fractions}
        results[model_name]={'observation_aware':aware,'model':model_meta,'metrics':summaries,
                             'gains_vs_zero_pct':gains,'coverage':realized}
        m0=base['final']['masked_mae']; m1=summaries[str(fractions[-1])]['final']['masked_mae']
        log(f'FINAL COVERAGE {model_name}: 0%={m0:.6f} 100%={m1:.6f} gain={gains[str(fractions[-1])]["final"]["masked_mae"]:+.3f}%')

    summary={'version':cfg['version'],'stage':a.stage,'run_tag':a.run_tag,
        'definition':'Deterministic nested restoration of actual finite benchmark-hidden historical cells; no amplitude interpolation',
        'base_checkpoint':str(base_path),'base_checkpoint_sha256':sha256_file(base_path),'base_checkpoint_source':source,
        'cache_identity':ident,'fractions':fractions,'restore_seed':restore_seed,
        'split':{'val_queries':len(ds),'dev_queries':len(dev_ids),'final_queries':len(final_ids),'seed':seed},
        'models':results,'elapsed_seconds':time.time()-t0}
    write_json(out/'summary.json',summary)
    log(f'SUMMARY={out/"summary.json"}')


if __name__=='__main__': main()
