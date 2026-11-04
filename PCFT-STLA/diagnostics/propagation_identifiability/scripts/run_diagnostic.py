#!/usr/bin/env python3
"""Gadi PCFT propagation identifiability diagnostic (TRAIN discovery, optional VAL confirmation).

Read-only. No TEST is opened, no model is trained. Small CSV/JSON output only.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

# This script is invoked without installing a Python package.
PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from diagnostic.physical import build_physical
from diagnostic.stability import compute_stability
from diagnostic.gain import choose_unique_windows, predict_cached, gain_for_candidate


def stamp(message):
    print(f'[{datetime.now().astimezone().isoformat(timespec="seconds")}] {message}', flush=True)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True,default=str),encoding='utf-8')


def repo_setup(repo,target):
    if not (repo/'Backbone').is_dir() or not (repo/'Base_SupportBench').is_dir():
        raise FileNotFoundError(f'Expected Backbone and Base_SupportBench under {repo}')
    os.environ['SPEED_LLM_RAG_ROOT']=str(repo)
    os.environ['TRAFFIDENT_TARGET']=target
    sys.path[:0] = [str(repo/'Base_SupportBench'),str(repo/'Backbone')]


def physical_and_stability(repo, target, args, out):
    from supportbench.dataset import SupportBench
    root=repo/'data/processed/traffident/base_supportbench'/target
    bench=SupportBench(args.dataset,root=root)
    assert bench.manifest['primary_channel']==target
    assert bench.N>1 and bench.H==12 and bench.P==12
    ei,ea,audit=build_physical(bench)
    split=bench.split_range('train')
    start,end=int(split['start']),int(split['end_exclusive'])
    cut=start+int((end-start)*args.discovery_fraction)
    # TRAIN-only discovery time range; later TRAIN and all VAL are unseen in discovery.
    x=np.asarray(bench.traffic[start:cut,:,bench.channel_names.index(target)],np.float32)
    m=np.asarray(bench.valid[start:cut,:,bench.channel_names.index(target)],bool)
    stamp(f'STABILITY_START target={target} nodes={bench.N} directed_candidates={ei.shape[1]} discovery=[{start},{cut})')
    stats=compute_stability(x,m,ei,blocks=args.blocks,max_lag=args.max_lag,
                            min_pairs=args.min_pairs,abs_corr=args.min_abs_corr,
                            sign_fraction=args.min_sign_fraction,
                            lag_fraction=args.min_lag_fraction,
                            direction_gap=args.min_direction_gap)
    del x,m
    cols={k:v for k,v in stats.items() if isinstance(v,np.ndarray) and v.ndim==1}
    cols['direction_class']=np.where(ea[:,6]>.5,'traffic_forward','disturbance_reverse')
    for lag in range(args.max_lag):cols[f'corr_lag{lag+1}']=stats['corr_by_lag'][:,lag]
    frame=pd.DataFrame(cols)
    frame.to_csv(out/'physical_direction_lag_candidates.csv',index=False,float_format='%.7g')
    stamp('STABILITY_DONE '+json.dumps({'candidates':len(frame), 'stable_lag':int(frame.stable_lag.sum()),
        'direction_and_lag':int(frame.stable_direction_lag.sum()),
        'forward':int((frame.direction_class=='traffic_forward').sum()),
        'reverse':int((frame.direction_class=='disturbance_reverse').sum())}))
    return bench,frame,audit,{'train_start':start,'train_end':end,'discovery_cut':cut,
         'benchmark_manifest_sha256':digest(bench.sg_dir/'manifest.json'),
         'nodes_sha256':digest(bench.sg_dir/'core/nodes.csv'),
         'adjacency_sha256':digest(bench.sg_dir/'graph/adjacency.npy')}


def load_official_stla(repo,target,bench,ckpt,cache_root,args):
    import torch
    from backbone.adapters import build_adapter
    from backbone.data import CachedSplit
    ck=torch.load(ckpt,map_location='cpu',weights_only=False)
    if ck.get('model')!='STLAformer' or int(ck.get('num_nodes',-1))!=bench.N:
        raise RuntimeError('Original STLA checkpoint model/node mismatch')
    if ck.get('target') not in (target,None):
        raise RuntimeError('Original STLA checkpoint target mismatch')
    if int(ck.get('seed',-1))!=args.seed:
        raise RuntimeError('Original STLA checkpoint seed mismatch')
    if ck.get('train_data_mode','incomplete')!='incomplete':
        raise RuntimeError('STLA checkpoint not trained in incomplete mode')
    ident=json.loads((cache_root/'cache_manifest.json').read_text())
    if ident.get('target_name')!=target:
        raise RuntimeError('Formal cache target mismatch')
    stats=ident['splits']['train']
    mean=float(stats.get('target_mean',stats.get('mean',ck['mean'])))
    std=float(stats.get('target_std',stats.get('std',ck['std'])))
    if not np.isclose([mean,std],[float(ck['mean']),float(ck['std'])],rtol=1e-4,atol=1e-4).all():
        raise RuntimeError('Backbone checkpoint normalization mismatch vs formal cache')
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    adapter=build_adapter('STLAformer',repo/'Backbone/vendor',bench.N,device,mean,std,
                          ck['model_config'],bench.adj,bench.nodes)
    adapter.model.load_state_dict(ck['state_dict'],strict=True)
    adapter.model.eval()
    train=CachedSplit(cache_root,'train',input_mode='incomplete'); train.root=cache_root
    val=CachedSplit(cache_root,'val',input_mode='incomplete') if args.confirm_val else None
    if val is not None:val.root=cache_root
    for split_name, ds in [('train',train),('val',val)]:
        if ds is None:continue
        if tuple(ds.x.shape[1:]) != (bench.H,bench.N):
            raise RuntimeError(f'{split_name} input shape mismatch: {ds.x.shape}')
        if tuple(ds.y_raw.shape[1:]) != (bench.P,bench.N):
            raise RuntimeError(f'{split_name} label shape mismatch: {ds.y_raw.shape}')
        if ds.valid.shape != ds.y_raw.shape:
            raise RuntimeError(f'{split_name} validity/label shape mismatch')
        if len(ds.meta) != len(ds):
            raise RuntimeError(f'{split_name} manifest length mismatch')
        mask_path=cache_root/f'{split_name}_history_observed.npy'
        if not mask_path.is_file():raise FileNotFoundError(mask_path)
        mask_shape=np.load(mask_path,mmap_mode='r').shape
        if mask_shape != ds.x.shape:raise RuntimeError(f'{split_name} mask shape mismatch {mask_shape}')
    return device,adapter,train,val,ident


def infer_gain(repo,target,bench,frame,ckpt,args,out,span):
    from backbone.shared_cache import cache_path
    from backbone.utils import load_yaml
    raw=load_yaml(repo/'Backbone/configs/backbone.yaml')
    def replace(x):
        if isinstance(x,dict):return {k:replace(v) for k,v in x.items()}
        if isinstance(x,list):return [replace(v) for v in x]
        if isinstance(x,str):return re.sub(r'TraffiDent_Net_\d\d',args.dataset,x)
        return x
    cfg=replace(raw)
    formal=Path(cache_path(cfg,'formal'))
    if not (formal/'cache_manifest.json').exists():
        raise FileNotFoundError(f'Formal TRAIN/VAL cache missing: {formal}')
    # A valid zero-candidate outcome should not spend GPU hours on STLA inference.
    qualified=frame[frame['stable_direction_lag'] & (frame['best_lag']>0)]
    if len(qualified)==0:
        empty_columns=['src','dst','lag','direction_class','corr','sign_fraction','lag_fraction',
                       'direction_gap','train_gain_pct','train_gain_vs_stla_pct','train_n',
                       'train_horizons','train_positive_groups','train_qualified','val_gain_pct',
                       'val_gain_vs_stla_pct','val_n','val_horizons','val_positive_groups',
                       'val_qualified','fully_confirmed']
        pd.DataFrame(columns=empty_columns).to_csv(out/'stla_incremental_gain_edges.csv',index=False)
        stamp(f'GAIN_SKIP_NO_CANDIDATES target={target} stable_direction_lag_candidates=0')
        return {'stable_direction_lag_candidates_tested':0,'train_gain_qualified':0,
                'val_gain_qualified':None,'confirmed_count':0,
                'confirmed_forward':0,'confirmed_reverse':0,
                'validation_confirmation':False,
                'validation_skipped_reason':'No stable direction/Lag candidates; 0 qualified by definition',
                'checkpoint_sha256':digest(ckpt),
                'formal_cache_manifest_sha256':digest(formal/'cache_manifest.json'),
                'fit_unique_windows':None,'heldout_unique_windows':None,'validation_unique_windows':None}
    device,adapter,train,val,identity=load_official_stla(repo,target,bench,ckpt,formal,args)
    ids=choose_unique_windows(train,args.max_train_views)
    if len(ids)==0:raise RuntimeError('No unique TRAIN windows in formal cache')
    starts=train.meta['window_start'].to_numpy(np.int64)[ids]
    cut=span['discovery_cut'];embargo=args.embargo
    holdout_start=span['train_start']+int((span['train_end']-span['train_start'])*args.holdout_fraction)
    fit_ids=ids[starts+bench.H+bench.P < cut-embargo]
    held_ids=ids[starts>=holdout_start+embargo]
    if len(fit_ids)<200 or len(held_ids)<100:
        raise RuntimeError(f'Insufficient unique TRAIN windows fit={len(fit_ids)} holdout={len(held_ids)}')
    stamp(f'STLA_START target={target} device={device} cache={formal} train_unique={len(ids)} '
          f'fit={len(fit_ids)} holdout={len(held_ids)} embargo={embargo}')
    fit=predict_cached(adapter,train,fit_ids,args.batch_size)
    held=predict_cached(adapter,train,held_ids,args.batch_size)
    if args.confirm_val:
        val_ids=choose_unique_windows(val,args.max_val_views)
        if len(val_ids)==0:raise RuntimeError('No unique VAL windows in formal cache')
        stamp(f'STLA_VAL_CONFIRM_START rows={len(val_ids)}')
        confirm=predict_cached(adapter,val,val_ids,args.batch_size)
    else:
        confirm=None
    import torch
    del adapter
    if torch.cuda.is_available():torch.cuda.empty_cache()
    results=[]
    stamp(f'GAIN_START target={target} stable_direction_lag_candidates={len(qualified)}')
    for count, row in enumerate(qualified.itertuples(index=False),1):
        source,dest,L=int(row.src),int(row.dst),int(row.best_lag)
        tr=gain_for_candidate(fit,held,source,dest,L,alpha=args.ridge_alpha,
                              min_fit=args.min_fit,min_eval=args.min_eval)
        cv=(gain_for_candidate(fit,confirm,source,dest,L,alpha=args.ridge_alpha,
                              min_fit=args.min_fit,min_eval=args.min_eval)
            if confirm is not None else None)
        def qualify(r):
            return bool(r and r['eligible'] and r['gain_vs_local_pct']>=args.min_gain_pct and
                        r['gain_vs_stla_pct']>0 and r['groups_with_data']>=3 and
                        r['positive_groups']>=3)
        ok_train=qualify(tr);ok_val=qualify(cv) if confirm is not None else None
        results.append(dict(src=source,dst=dest,lag=L,direction_class=row.direction_class,
                            corr=row.corr,sign_fraction=row.sign_fraction,lag_fraction=row.lag_fraction,
                            direction_gap=row.direction_gap,
                            train_gain_pct=tr['gain_vs_local_pct'],
                            train_gain_vs_stla_pct=tr['gain_vs_stla_pct'],
                            train_n=tr['n'],train_horizons=tr['horizons'],
                            train_positive_groups=tr['positive_groups'],
                            train_qualified=ok_train,
                            val_gain_pct=cv['gain_vs_local_pct'] if cv else None,
                            val_gain_vs_stla_pct=cv['gain_vs_stla_pct'] if cv else None,
                            val_n=cv['n'] if cv else None,
                            val_horizons=cv['horizons'] if cv else None,
                            val_positive_groups=cv['positive_groups'] if cv else None,
                            val_qualified=ok_val,
                            fully_confirmed=ok_train and (ok_val if confirm is not None else True)))
        if count%100==0:
            stamp(f'GAIN_PROGRESS edges={count}/{len(qualified)} train_qualified={sum(x["train_qualified"] for x in results)}')
    # Preserve machine-readable column headers even if discovery finds zero candidates.
    empty_columns=['src','dst','lag','direction_class','corr','sign_fraction','lag_fraction',
                   'direction_gap','train_gain_pct','train_gain_vs_stla_pct','train_n',
                   'train_horizons','train_positive_groups','train_qualified','val_gain_pct',
                   'val_gain_vs_stla_pct','val_n','val_horizons','val_positive_groups',
                   'val_qualified','fully_confirmed']
    df=pd.DataFrame(results,columns=empty_columns)
    df.to_csv(out/'stla_incremental_gain_edges.csv',index=False,float_format='%.7g')
    nconfirmed=int(df['fully_confirmed'].sum()) if len(df) else 0
    return {'stable_direction_lag_candidates_tested':int(len(qualified)),
            'train_gain_qualified':int(df['train_qualified'].sum()) if len(df) else 0,
            'val_gain_qualified':int(df['val_qualified'].sum()) if (len(df) and confirm is not None) else None,
            'confirmed_count':nconfirmed,
            'confirmed_forward':int(((df['direction_class']=='traffic_forward')&df['fully_confirmed']).sum()) if len(df) else 0,
            'confirmed_reverse':int(((df['direction_class']=='disturbance_reverse')&df['fully_confirmed']).sum()) if len(df) else 0,
            'comparison':'STLA+local+source vs STLA+local on identical source-observed rows',
            'anchor':'frozen official STLAformer, incomplete s40',
            'horizons':'1..lag only; H>lag needs unobserved future source and is excluded',
            'validation_confirmation': bool(confirm is not None),
            'checkpoint_sha256':digest(ckpt),
            'formal_cache_manifest_sha256':digest(formal/'cache_manifest.json'),
            'fit_unique_windows':len(fit_ids),'heldout_unique_windows':len(held_ids),
            'validation_unique_windows':len(confirm.starts) if confirm is not None else None}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',default=os.environ.get('SPEED_LLM_RAG_ROOT','.'),help='Speed_LLM_RAG root')
    ap.add_argument('--dataset',default='TraffiDent_Net_06')
    ap.add_argument('--target',choices=['speed','flow'],required=True)
    ap.add_argument('--checkpoint',help='Official STLAformer Backbone best.pt; required for --phase all')
    ap.add_argument('--seed',type=int,default=10)
    ap.add_argument('--phase',choices=['preflight','stats','all'],default='all')
    ap.add_argument('--out',default=None)
    ap.add_argument('--cpu',action='store_true',help='Force CPU inference (slow)')
    ap.add_argument('--preflight-gpu',action='store_true',help='For preflight only: use actual CUDA model inference on allocated GPU')
    ap.add_argument('--confirm-val',action='store_true',help='One-time confirmation on untouched VAL, never used for fitting')
    ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--max-train-views',type=int,default=12000)
    ap.add_argument('--max-val-views',type=int,default=2500)
    ap.add_argument('--discovery-fraction',type=float,default=.65)
    ap.add_argument('--holdout-fraction',type=float,default=.75)
    ap.add_argument('--embargo',type=int,default=48)
    ap.add_argument('--blocks',type=int,default=6)
    ap.add_argument('--max-lag',type=int,default=6)
    ap.add_argument('--min-pairs',type=int,default=64)
    ap.add_argument('--min-abs-corr',type=float,default=.15)
    ap.add_argument('--min-sign-fraction',type=float,default=.75)
    ap.add_argument('--min-lag-fraction',type=float,default=.66)
    ap.add_argument('--min-direction-gap',type=float,default=.03)
    ap.add_argument('--ridge-alpha',type=float,default=3.0)
    ap.add_argument('--min-fit',type=int,default=60)
    ap.add_argument('--min-eval',type=int,default=30)
    ap.add_argument('--min-gain-pct',type=float,default=.2)
    args=ap.parse_args()
    repo=Path(args.repo).resolve();repo_setup(repo,args.target)
    if args.preflight_gpu and (args.phase!='preflight' or args.cpu):
        raise ValueError('--preflight-gpu requires --phase preflight without --cpu')
    if not .4<args.discovery_fraction<args.holdout_fraction<.9:
        raise ValueError('Invalid chronological fractions')
    if not 1 <= args.max_lag <= 11:
        raise ValueError('max_lag must be 1..11 for a 12-step history')
    if args.blocks < 3 or args.min_pairs < 2 or args.batch_size < 1:
        raise ValueError('blocks>=3, min_pairs>=2 and batch_size>=1 required')
    if args.max_train_views < 1 or args.max_val_views < 1:
        raise ValueError('max_train_views and max_val_views must be >=1')
    if args.embargo < 0 or args.min_fit < 2 or args.min_eval < 2:
        raise ValueError('Invalid embargo/minimum sample requirements')
    if args.phase in ('preflight','all') and not args.checkpoint:ap.error('--checkpoint required for --phase preflight/all')
    if args.phase=='all' and not args.cpu:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Full STLA residual inference needs CUDA; run from GPU PBS or use --cpu explicitly')
    output=Path(args.out or (repo/'PCFT-STLA_diagnostics'/'propagation_identifiability'/args.dataset/args.target))
    output.mkdir(parents=True,exist_ok=True)
    stamp('DIAGNOSTIC_START '+json.dumps({'target':args.target,'phase':args.phase,'dataset':args.dataset,'out':str(output)}))
    st=time.time()
    # Published full summary is valid only for a completed current full run.
    if args.phase=='all':
        (output/'SUMMARY.json').unlink(missing_ok=True)
    if args.phase=='preflight':
        from supportbench.dataset import SupportBench
        bench=SupportBench(args.dataset,root=repo/'data/processed/traffident/base_supportbench'/args.target)
        if bench.manifest.get('primary_channel')!=args.target:raise RuntimeError('Wrong benchmark target')
        ei,ea,audit=build_physical(bench)
        stamp(f'PREFLIGHT_GRAPH nodes={bench.N} physical_edges={ei.shape[1]}')
        # Exercise the *real* raw TRAIN -> direction/Lag discovery path before
        # requesting a GPU. This bounded CPU smoke catches graph/array/edge
        # alignment failures that an STLA forward alone would never test.
        train_range=bench.split_range('train')
        smoke_start=int(train_range['start'])
        smoke_end=min(int(train_range['end_exclusive']),smoke_start+768)
        if smoke_end-smoke_start < (args.max_lag+2)*6:
            raise RuntimeError('Insufficient TRAIN timesteps for the stability smoke')
        target_channel=bench.channel_names.index(args.target)
        smoke_x=np.asarray(bench.traffic[smoke_start:smoke_end,:,target_channel],np.float32)
        smoke_m=np.asarray(bench.valid[smoke_start:smoke_end,:,target_channel],bool)
        smoke_stats=compute_stability(smoke_x,smoke_m,ei,blocks=3,max_lag=args.max_lag,
                                      min_pairs=min(args.min_pairs,16),
                                      abs_corr=args.min_abs_corr,
                                      sign_fraction=args.min_sign_fraction,
                                      lag_fraction=args.min_lag_fraction,
                                      direction_gap=args.min_direction_gap)
        if smoke_stats['corr_by_lag'].shape != (ei.shape[1],args.max_lag):
            raise RuntimeError('Stability smoke returned incorrect edge/lag dimensions')
        stamp(f'PREFLIGHT_STATS_PASS target={args.target} '
              f'edges={ei.shape[1]} lags={args.max_lag} timesteps={len(smoke_x)}')
        from backbone.shared_cache import cache_path
        from backbone.utils import load_yaml
        cfg=load_yaml(repo/'Backbone/configs/backbone.yaml')
        cache=Path(cache_path(cfg,'formal'))
        ckpt=Path(args.checkpoint);ckpt=ckpt if ckpt.is_absolute() else repo/ckpt
        if args.preflight_gpu:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('--preflight-gpu requested but CUDA is unavailable')
            args.cpu=False
        else:
            args.cpu=True
        device,adapter,train,val,identity=load_official_stla(repo,args.target,bench,ckpt,cache,args)
        if args.preflight_gpu and device.type!='cuda':
            raise RuntimeError('GPU preflight did not place STLA on CUDA')
        a=predict_cached(adapter,train,np.array([0],dtype=np.int64),batch_size=1)
        from diagnostic.gain import audit_forecast_arrays
        # Uses the SAME feature builder as the full gain stage, and verifies all
        # H1/Hlag index conventions; no stale hard-coded feature dimension.
        train_contract=audit_forecast_arrays(a, split='TRAIN', max_lag=args.max_lag)
        stamp('PREFLIGHT_TRAIN_FEATURES_PASS '+json.dumps(train_contract,sort_keys=True))
        val_contract=None
        if args.confirm_val:
            if val is None or len(val)==0:raise RuntimeError('Empty VAL split')
            b=predict_cached(adapter,val,np.array([0],dtype=np.int64),batch_size=1)
            val_contract=audit_forecast_arrays(b, split='VAL', max_lag=args.max_lag)
            stamp('PREFLIGHT_VAL_FEATURES_PASS '+json.dumps(val_contract,sort_keys=True))
            stamp(f'PREFLIGHT_VAL_PASS target={args.target} shape={tuple(b.pred.shape)}')
        write_json(output/'PREFLIGHT.json',dict(target=args.target,nodes=bench.N,physical_edges=int(ei.shape[1]),
            checkpoint_sha256=digest(ckpt),cache_manifest_sha256=digest(cache/'cache_manifest.json'),
            prediction_shape=list(a.pred.shape),prediction_finite=bool(np.isfinite(a.pred).all()),
            train_feature_contract=train_contract,val_feature_contract=val_contract))
        stamp(f'PREFLIGHT_PASS target={args.target} device={device.type} shape={tuple(a.pred.shape)}')
        return
    bench,frame,audit,span=physical_and_stability(repo,args.target,args,output)
    summary={'schema':'pcft_propagation_diagnostic_v1','target':args.target,'dataset':args.dataset,
             'discovery_data':'raw_valid TRAIN readings: OPTIMISTIC upper bound; s40 is assessed separately',
             'causality_claim':False,'physical_directed_candidates':len(frame),
             'discovery_stable_lag_edges':int(frame['stable_lag'].sum()),
             'discovery_direction_and_lag_edges':int(frame['stable_direction_lag'].sum()),
             'discovery_direction_and_lag_forward':int((frame.stable_direction_lag & (frame.direction_class=='traffic_forward')).sum()),
             'discovery_direction_and_lag_reverse':int((frame.stable_direction_lag & (frame.direction_class=='disturbance_reverse')).sum()),
             'directionality_operational_definition':'abs(r_direction)-abs(r_reverse)>=min_direction_gap; predictive lead asymmetry not physical causality',
             'lag_stability_operational_definition':'block sign and best-lag +/-1 agreement on early TRAIN only',
             'time_ranges':span,'physical_audit':{k:v for k,v in audit.items() if k!='node_ids'},
             'thresholds':{k:v for k,v in vars(args).items() if k.startswith('min_') or k in ('blocks','max_lag','embargo','ridge_alpha','discovery_fraction','holdout_fraction')},
             'run_time_seconds':0}
    if args.phase=='all':
        ckpt=Path(args.checkpoint);ckpt=ckpt if ckpt.is_absolute() else repo/ckpt
        if not ckpt.is_file():raise FileNotFoundError(ckpt)
        summary['stla_incremental']=infer_gain(repo,args.target,bench,frame,ckpt,args,output,span)
    summary['run_time_seconds']=round(time.time()-st,2)
    # A CPU-only stats run must never replace a previously successful full report.
    dest = 'SUMMARY_STATS_ONLY.json' if args.phase=='stats' else 'SUMMARY.json'
    write_json(output/dest,summary)
    stamp('DIAGNOSTIC_COMPLETE '+json.dumps({k:v for k,v in summary.items() if k in ('physical_directed_candidates','discovery_stable_lag_edges','discovery_direction_and_lag_edges','stla_incremental','run_time_seconds')},default=str))
    stamp(f'OUTPUT_DIRECTORY {output}')

if __name__=='__main__':main()
