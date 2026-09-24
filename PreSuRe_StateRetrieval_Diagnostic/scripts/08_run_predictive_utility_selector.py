#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys, time, math
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts._common import setup
from srd.contract import assert_cache_ready
from srd.backbone import load_patchstg_adapter
from srd.data import CacheView
from srd.evaluate import predict_raw_batched
from srd.metrics import masked_mae, improvement_pct
from srd.patch_layout import extract_patch_layout
from srd.patch_transport import candidate_proposal, inject_patch
from srd.utility_selector import (
    DEFAULT_MODELS, parse_int_list, parse_str_list, add_inference_features, feature_columns,
    make_model, fit_model, predict_scores, expanding_time_folds, blocked_folds,
    tune_gate_and_cap, apply_decision_rule, action_metrics, json_safe,
)
from srd.utils import write_json


def log(msg): print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}",flush=True)

def gpu(device):
    if getattr(device,'type','cpu')!='cuda' or not torch.cuda.is_available(): return 'device=cpu'
    try:
        return f"gpu_alloc={torch.cuda.memory_allocated(device)/2**30:.2f}GB gpu_reserved={torch.cuda.memory_reserved(device)/2**30:.2f}GB"
    except Exception: return 'device=cuda'


def _compose(q, selected, train, layout, L, transport, cfg, cache):
    x=np.asarray(q["x_obs"],np.float32).copy(); used=[]
    if selected is None or len(selected)==0: return x,used
    # Do not use itertuples() here: pandas renames columns that begin with
    # an underscore (e.g. `_score`) to positional field names, so row._score
    # is not stable across pandas versions. Dictionary records preserve the
    # exact selector column names.
    ordered = selected.sort_values('_score', ascending=False)
    for row in ordered.to_dict(orient='records'):
        cid=int(row['candidate_id']); pid=int(row['patch_id'])
        cand=cache.get(cid)
        if cand is None: cand=train.row(cid); cache[cid]=cand
        qexec=dict(q); qexec['restorable']=np.asarray(q['artificial_missing'],bool)
        p,f=candidate_proposal(qexec,cand,layout.groups[pid],L,transport,cfg)
        if np.any(f):
            x=inject_patch(x,p,f); used.append((pid,cid,int(row['rank']),float(row['_score']),float(row['forecast_utility'])))
    return x,used


def _evaluate_completed(adapter,ck,queries,completed,device,batch_size):
    if not queries: return np.empty((0,),np.float64)
    X=np.stack(completed).astype(np.float32)
    tod=np.stack([q['tod_hist'] for q in queries]).astype(np.int64)
    dow=np.stack([q['dow_hist'] for q in queries]).astype(np.int64)
    pred=predict_raw_batched(adapter,ck,X,tod,dow,device,batch_size=batch_size)
    y=np.stack([q['y_raw'] for q in queries]).astype(np.float32)
    fv=np.stack([q['future_valid'] for q in queries]).astype(bool)
    dm=np.stack([q['direct_mask'] for q in queries]).astype(bool)
    return masked_mae(pred,y,fv,dm)


def _oracle_lookup(run_dir:Path):
    p=run_dir/'forecast_utility_oracle'/'forecast_oracle_by_k.csv'
    if not p.exists(): return {}
    d=pd.read_csv(p)
    d=d[(d['mode']=='patch_subset_prefix_forecast_oracle')]
    out={}
    for row in d.itertuples(index=False): out[(int(row.recent_steps),int(row.K))]=float(row.forecast_gain_pct)
    return out


def main():
    ap=argparse.ArgumentParser(description='v0.7 Predictive-Utility Gate + Reranker')
    ap.add_argument('--run-dir',required=True)
    ap.add_argument('--stage',default='development',choices=['development','formal'])
    ap.add_argument('--checkpoint',default=None)
    ap.add_argument('--dataset',default=None,help='utility_action_dataset.csv.gz; defaults under run-dir')
    ap.add_argument('--models',default=','.join(DEFAULT_MODELS))
    ap.add_argument('--scopes',default='1,3,6,12')
    ap.add_argument('--max-k',type=int,default=50)
    ap.add_argument('--protocol',default='expanding',choices=['expanding','blocked'])
    ap.add_argument('--cv-blocks',type=int,default=6)
    ap.add_argument('--patch-caps',default='1,2,4,6,8,12,16')
    ap.add_argument('--transport',default='affine_alignment')
    ap.add_argument('--eval-batch-size',type=int,default=64)
    ap.add_argument('--seed',type=int,default=10)
    a=ap.parse_args()

    cfg=setup(a.stage); assert_cache_ready(cfg,a.stage)
    rd=Path(a.run_dir).resolve(); out=rd/'predictive_utility_selector'; out.mkdir(parents=True,exist_ok=True)
    data_path=Path(a.dataset).resolve() if a.dataset else out/'utility_action_dataset.csv.gz'
    if not data_path.exists(): raise FileNotFoundError(data_path)
    df=pd.read_csv(data_path); df=add_inference_features(df)
    scopes=parse_int_list(a.scopes,[1,3,6,12]); models=parse_str_list(a.models,DEFAULT_MODELS); caps=parse_int_list(a.patch_caps,[1,2,4,6,8,12,16])
    df=df[df['recent_steps'].isin(scopes)&(df['rank']<=int(a.max_k))].copy()
    feats=feature_columns(df)
    qorder=df[['query_pos','query_id']].drop_duplicates().sort_values('query_pos').query_id.astype(int).tolist()
    folds=list(expanding_time_folds(qorder,a.cv_blocks) if a.protocol=='expanding' else blocked_folds(qorder,a.cv_blocks))
    if not folds: raise RuntimeError('No CV folds')

    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    adapter,cp,ck,source=load_patchstg_adapter(cfg,a.stage,device,a.checkpoint)
    layout=extract_patch_layout(adapter.model,ck,backbone_root=cfg['project']['backbone_root'])
    train,val=CacheView(cfg,a.stage,'train'),CacheView(cfg,a.stage,'val')
    oracle=_oracle_lookup(rd)
    log('='*88); log('Predictive-Utility Gate + Reranker v0.7.2 START')
    log(f'run={rd} rows={len(df)} queries={len(qorder)} features={len(feats)} models={models}')
    log(f'protocol={a.protocol} folds={len(folds)} scopes={scopes} max_k={a.max_k} patch_caps={caps}')
    log(f'checkpoint={cp} source={source} transport={a.transport} {gpu(device)}'); log('='*88)

    all_fold=[]; all_query=[]; all_action=[]; start=time.time()
    for mi,mname in enumerate(models,1):
        log(f'MODEL {mi}/{len(models)} {mname} START')
        for fold,fit_q,cal_q,test_q in folds:
            fs=time.time(); fit_df=df[df.query_id.isin(fit_q)].copy(); cal_df=df[df.query_id.isin(cal_q)].copy(); test_df=df[df.query_id.isin(test_q)].copy()
            log(f'MODEL={mname} FOLD={fold} fitQ={len(fit_q)} calQ={len(cal_q)} testQ={len(test_q)} fitRows={len(fit_df)}')
            spec=fit_model(make_model(mname,a.seed+fold),fit_df,feats)
            cal_scores=predict_scores(spec,cal_df,feats); test_scores=predict_scores(spec,test_df,feats)
            am=action_metrics(test_df,test_scores); am.update({'model':mname,'fold':fold,'protocol':a.protocol}); all_action.append(am)
            # map global test scores by dataframe index for scope slicing
            test_score_series=pd.Series(test_scores,index=test_df.index)
            cal_score_series=pd.Series(cal_scores,index=cal_df.index)
            for L in scopes:
                calL=cal_df[cal_df.recent_steps==L].copy(); testL=test_df[test_df.recent_steps==L].copy()
                # Do not skip an OOF scope merely because the calibration or test action
                # table is empty. An empty action table means 'abstain everywhere' for that
                # partition; every test query must still be evaluated at its untouched baseline.
                gate=tune_gate_and_cap(calL,cal_score_series.loc[calL.index].to_numpy(),a.max_k,caps)
                selected=apply_decision_rule(testL,test_score_series.loc[testL.index].to_numpy(),a.max_k,gate['threshold'],gate['max_patches'])
                sel_groups={int(q):g.copy() for q,g in selected.groupby('query_id',sort=False)} if len(selected) else {}
                queries=[]; completed=[]; qmeta=[]; cand_cache={}
                for qid in test_q:
                    qid=int(qid); q=val.row(qid); sel=sel_groups.get(qid)
                    x,used=_compose(q,sel,train,layout,L,a.transport,cfg,cand_cache)
                    queries.append(q); completed.append(x); qmeta.append((qid,used))
                maes=_evaluate_completed(adapter,ck,queries,completed,device,a.eval_batch_size)
                # IMPORTANT: scope-specific action tables may omit queries with no eligible
                # patch at this recent_steps value. Those queries still belong to the OOF
                # end-to-end evaluation and must contribute their untouched PatchSTG baseline.
                # Therefore baseline lookup must come from the full fold test table rather than
                # testL, which contains only queries that have at least one candidate action.
                base_map=test_df.groupby('query_id',sort=False).baseline_forecast_mae.first()
                missing_baseline=[int(q) for q in test_q if int(q) not in base_map.index]
                if missing_baseline:
                    raise RuntimeError(f'Missing baseline_forecast_mae for test queries: {missing_baseline[:10]}')
                for (qid,used),mae in zip(qmeta,maes):
                    base=float(base_map.loc[qid]); indiv=[u[-1] for u in used]
                    all_query.append({
                        'model':mname,'protocol':a.protocol,'fold':int(fold),'query_id':qid,'recent_steps':int(L),
                        'baseline_forecast_mae':base,'realized_forecast_mae':float(mae),
                        'realized_utility':base-float(mae),'realized_gain_pct':improvement_pct(base,float(mae)),
                        'selected_patches':len(used),'selected_individual_beneficial':int(sum(v>0 for v in indiv)),
                        'selected_individual_harmful':int(sum(v<-0.01 for v in indiv)),
                        'selected_individual_mean_utility':float(np.mean(indiv)) if indiv else 0.0,
                        'threshold':float(gate['threshold']),'max_patches':int(gate['max_patches']),
                    })
                qpart=[r for r in all_query if r['model']==mname and r['fold']==fold and r['recent_steps']==L]
                b=float(np.mean([r['baseline_forecast_mae'] for r in qpart])); mm=float(np.mean([r['realized_forecast_mae'] for r in qpart]))
                selrows=selected
                prec=float((selrows.forecast_utility>0).mean()) if len(selrows) else np.nan
                hir=float((selrows.forecast_utility<-0.01).mean()) if len(selrows) else 0.0
                fr={
                    'model':mname,'protocol':a.protocol,'fold':int(fold),'recent_steps':int(L),
                    'fit_queries':int(len(fit_q)),'calib_queries':int(len(cal_q)),'test_queries':int(len(test_q)),
                    'threshold':float(gate['threshold']),'max_patches':int(gate['max_patches']),
                    'calib_additive_utility':float(gate['calib_additive_utility']),
                    'baseline_forecast_mae':b,'realized_forecast_mae':mm,'realized_gain_pct':improvement_pct(b,mm),
                    'intervention_query_rate':float(np.mean([r['selected_patches']>0 for r in qpart])),
                    'mean_selected_patches':float(np.mean([r['selected_patches'] for r in qpart])),
                    'selected_patch_precision':prec,'harmful_selected_patch_rate':hir,
                    'oracle_reference_pct':oracle.get((int(L),int(a.max_k)),np.nan),
                }
                all_fold.append(fr)
                log(f"MODEL={mname} FOLD={fold} recent{L} gain={fr['realized_gain_pct']:+.3f}% IR={fr['intervention_query_rate']:.3f} patches={fr['mean_selected_patches']:.2f} precision={prec:.3f} harmful={hir:.3f} tau={gate['threshold']:.5g} cap={gate['max_patches']} elapsed={time.time()-fs:.1f}s")
            log(f'MODEL={mname} FOLD={fold} DONE elapsed={time.time()-fs:.1f}s {gpu(device)}')
        log(f'MODEL {mname} DONE total_elapsed={time.time()-start:.1f}s')

    qdf=pd.DataFrame(all_query); fdf=pd.DataFrame(all_fold); adf=pd.DataFrame(all_action)
    qdf.to_csv(out/'oof_query_results.csv',index=False); fdf.to_csv(out/'fold_metrics.csv',index=False); adf.to_csv(out/'action_metrics.csv',index=False)
    rows=[]
    for (m,L),g in qdf.groupby(['model','recent_steps'],sort=True):
        b=float(g.baseline_forecast_mae.mean()); mm=float(g.realized_forecast_mae.mean()); gain=improvement_pct(b,mm)
        folds_m=fdf[(fdf.model==m)&(fdf.recent_steps==L)]
        o=oracle.get((int(L),int(a.max_k)),np.nan)
        rows.append({
            'model':m,'protocol':a.protocol,'recent_steps':int(L),'oof_queries':int(g.query_id.nunique()),
            'baseline_forecast_mae':b,'realized_forecast_mae':mm,'realized_gain_pct':gain,
            'oracle_reference_pct':o,'oracle_conversion_ratio':gain/o if np.isfinite(o) and o>1e-12 else np.nan,
            'intervention_query_rate':float((g.selected_patches>0).mean()),'mean_selected_patches':float(g.selected_patches.mean()),
            'selected_individual_precision':float(g.selected_individual_beneficial.sum()/max(g.selected_patches.sum(),1)),
            'harmful_selected_patch_rate':float(g.selected_individual_harmful.sum()/max(g.selected_patches.sum(),1)),
            'mean_threshold':float(folds_m.threshold.mean()) if len(folds_m) else np.nan,
            'mean_max_patches':float(folds_m.max_patches.mean()) if len(folds_m) else np.nan,
        })
    sm=pd.DataFrame(rows).sort_values(['recent_steps','realized_gain_pct'],ascending=[True,False])
    sm.to_csv(out/'model_comparison.csv',index=False)
    primary_scope=int(cfg.get('predictive_utility_selector',{}).get('primary_scope',12))
    primary=sm[sm.recent_steps==primary_scope].sort_values('realized_gain_pct',ascending=False)
    best=primary.iloc[0].to_dict() if len(primary) else (sm.sort_values('realized_gain_pct',ascending=False).iloc[0].to_dict() if len(sm) else None)
    payload={
        'version':'0.7.2','protocol':a.protocol,'cv_blocks':int(a.cv_blocks),'max_k':int(a.max_k),'scopes':scopes,'models':models,
        'feature_count':len(feats),'features':feats,'checkpoint':str(cp),'checkpoint_source':source,
        'primary_scope':primary_scope,'best_primary_model':json_safe(best),
        'target_gain_pct':float(cfg.get('predictive_utility_selector',{}).get('target_gain_pct',3.0)),
        'strong_target_gain_pct':float(cfg.get('predictive_utility_selector',{}).get('strong_target_gain_pct',5.0)),
        'results':json_safe(rows),
        'leakage_contract':'fit/calibration utility labels precede each expanding-CV test block; model features contain no hidden-history truth or future-target values',
    }
    write_json(out/'summary.json',payload)
    log('='*88); log('V0.7 MODEL COMPARISON'); print(sm.to_string(index=False),flush=True)
    log('V0.7 SUMMARY'); print(json.dumps(json_safe({k:v for k,v in payload.items() if k not in ['features','results']}),indent=2),flush=True)
    log(f'TOTAL_ELAPSED={time.time()-start:.1f}s {gpu(device)}'); log(f'SUMMARY={out/"summary.json"}'); log('Predictive-Utility Gate + Reranker v0.7.2 COMPLETE'); log('='*88)

if __name__=='__main__': main()
