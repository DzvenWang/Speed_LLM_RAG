"""Paired, query-clustered FARE counterfactual diagnostics; no model training."""
from __future__ import annotations
from collections import defaultdict
import math
import numpy as np
from .data import DiagnosticShard

def _finite_mean(x):
    arr=np.asarray(x,dtype=float)
    arr=arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else None

def query_errors(truth, pred, mask):
    if not np.all(np.isfinite(pred[mask])) or not np.all(np.isfinite(truth[mask])):
        raise ValueError('Non-finite y_true or prediction inside valid mask')
    reduce_axis=tuple(range(1,truth.ndim))
    valid=mask.sum(axis=reduce_axis)
    if np.any(valid==0): raise ValueError('At least one query has zero valid targets')
    d=np.where(mask,pred-truth,0)
    mae=np.sum(np.abs(d),axis=reduce_axis)/valid
    mse=np.sum(d*d,axis=reduce_axis)/valid
    return mae,mse,valid

def corr(a,b):
    if a is None or b is None:return None
    a=np.asarray(a,dtype=float).ravel();b=np.asarray(b,dtype=float).ravel()
    if len(a)!=len(b):return None
    k=np.isfinite(a)&np.isfinite(b)
    if k.sum()<3 or np.std(a[k])<1e-12 or np.std(b[k])<1e-12:return None
    return float(np.corrcoef(a[k],b[k])[0,1])

def _scalar_feature(x,n):
    x=np.asarray(x,dtype=float)
    if x.ndim == 0:return None
    if x.size==n:return x.reshape(n)
    if x.ndim and x.shape[0]==n:return np.nanmean(x.reshape(n,-1),axis=1)
    return None

def _ci_mean(x,seed=10,n_boot=400):
    x=np.asarray(x,dtype=float)
    if len(x)==0:return [None,None]
    rng=np.random.default_rng(seed)
    estimates=np.empty(n_boot)
    # Paired bootstrap resamples queries, never individual nodes/timesteps.
    for i in range(n_boot):estimates[i]=np.mean(x[rng.integers(0,len(x),len(x))])
    return [float(v) for v in np.quantile(estimates,[.025,.975])]

def evaluate_shard(shard:DiagnosticShard,seed=10,n_boot=400):
    per={}; mode_stat={}; t=shard.truth;m=shard.mask
    for key,pred in shard.modes.items():
        qmae,qmse,valid=query_errors(t,pred,m)
        per[key]=(qmae,qmse,valid)
        mode_stat[key]={
            'MAE':float(np.sum(qmae*valid)/np.sum(valid)),
            'RMSE':float(np.sqrt(np.sum(qmse*valid)/np.sum(valid))),
            'query_mean_MAE':float(np.mean(qmae)),
        }
    base=per['no_evidence'][0];full=per['full'][0];gain=base-full
    diag={
        'target':shard.metadata['target'],'severity':int(shard.metadata['severity']),
        'failure_family':shard.metadata['failure_family'],'split':shard.metadata['split'],
        'checkpoint_sha256':shard.metadata['checkpoint_sha256'],'seed':shard.metadata['seed'],
        'queries':int(shard.n), 'modes':mode_stat,
        'gain_MAE_query_mean':float(np.mean(gain)),
        'gain_MAE_CI95':_ci_mean(gain,seed,n_boot),
        'harm_rate':float(np.mean(gain < -1e-10)),
        'harm_mean':_finite_mean(-gain[gain<0]),
        'beneficial_rate':float(np.mean(gain > 1e-10)),
        'worst_query_harm':float(np.max(-gain)),
        'unsupported_modes':[],
    }
    diag['missing_recommended_modes']=[k for k in ('hidden_only','forecast_only','null_evidence','no_gate',
                                'similarity_only','random','shuffled','no_support_demand',
                                'no_direction','no_propagation_alignment','no_set_interaction',
                                'no_counterfactual_utility') if k not in per]
    if 'null_evidence' in shard.modes:
        a=shard.modes['null_evidence'];b=shard.modes['no_evidence']
        diag['null_path_max_abs_diff']=float(np.max(np.abs(a[m]-b[m])))
        diag['null_path_equivalent_1e-5']=bool(diag['null_path_max_abs_diff'] <= 1e-5)
    if shard.candidate_predictions is not None:
        pred=shard.candidate_predictions
        if pred.shape[1] > 0:
            qm=[]
            for j in range(pred.shape[1]):qm.append(query_errors(t,pred[:,j],m)[0])
            oracle_mae=np.min(np.stack(qm,axis=1),axis=1)
            potential=base-oracle_mae
            diag['oracle_best_candidate_gain']=float(np.mean(potential))
            diag['oracle_selected_slot_limited']=True
            diag['oracle_positive_rate']=float(np.mean(potential>1e-10))
            diag['realized_gain_vs_oracle']=float(np.mean(gain))
            diag['oracle_capture_ratio']=float(np.mean(gain)/np.mean(potential)) if np.mean(potential)>1e-8 else None
            diag['candidate_count']=int(pred.shape[1])
            diag['oracle_scope']=shard.metadata.get('oracle_scope','unknown_do_not_claim_global_pool')
            diag['oracle_CI95']=_ci_mean(potential,seed,n_boot)
            diag['demand_vs_oracle_utility_r']=corr(_scalar_feature(shard.extras['demand'],shard.n),potential) if 'demand' in shard.extras else None
            diag['missingness_vs_oracle_utility_r']=corr(_scalar_feature(shard.extras['mask_ratio'],shard.n),potential) if 'mask_ratio' in shard.extras else None
    for k in ['alpha','null_weight','demand','mask_ratio','propagation']:
        if k in shard.extras:
            v=_scalar_feature(shard.extras[k],shard.n)
            if v is not None:
                diag[k+'_mean']=_finite_mean(v)
                diag[k+'_vs_realized_gain_r']=corr(v,gain)
    for k in ('hidden_only','forecast_only','similarity_only','random','shuffled', 'no_gate',
              'no_support_demand','no_direction','no_propagation_alignment','no_set_interaction','no_counterfactual_utility'):
        if k in per:
            diff=base-per[k][0]
            diag[k+'_gain_query_mean']=float(np.mean(diff))
            diag[k+'_vs_full_gain_query_mean']=float(np.mean(per[k][0]-full))
    return diag

def summarize(items):
    out={}
    for target in ('speed','flow'):
        xs=[x for x in items if x['target']==target]
        if not xs:continue
        out[target]={
            'shards':len(xs),'queries':sum(x['queries'] for x in xs),
            'severity_coverage':sorted(set(x['severity'] for x in xs)),
            'failure_family_coverage':sorted(set(x['failure_family'] for x in xs)),
            'mean_shard_full_MAE':_finite_mean([x['modes']['full']['MAE'] for x in xs]),
            'mean_shard_no_evidence_MAE':_finite_mean([x['modes']['no_evidence']['MAE'] for x in xs]),
            'mean_shard_gain':_finite_mean([x['gain_MAE_query_mean'] for x in xs]),
            'mean_shard_harm_rate':_finite_mean([x['harm_rate'] for x in xs]),
            'mean_oracle_capture_ratio':_finite_mean([x.get('oracle_capture_ratio') for x in xs if x.get('oracle_capture_ratio') is not None]),
            'null_path_failed_shards':[x['severity'] for x in xs if x.get('null_path_equivalent_1e-5') is False],
        }
    return out
