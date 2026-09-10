#!/usr/bin/env python3
"""Build the legacy v1 hand-crafted Support Opportunity diagnostic.

The script is retained for reproducibility only.  Since v1.2.3 its
``predictive_context_loss`` column uses the formal Chapter-3 network mean; the
old q-quantile is emitted separately as ``legacy_context_loss_quantile``.
Formal experiments should use Support Opportunity v2 instead.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from supportbench.config import load_config
from supportbench.dataset import SupportBench
from supportbench.utils import ensure_dir, log, write_json


def entropy_score(scores: np.ndarray, temperature: float) -> float:
    s=np.asarray(scores,float); s=s[np.isfinite(s)]
    if len(s)<=1: return 0.0
    z=(s-np.max(s))/max(float(temperature),1e-6); p=np.exp(z); p=p/max(float(p.sum()),1e-12)
    h=-float(np.sum(p*np.log(np.maximum(p,1e-12))))
    return h/max(float(np.log(len(p))),1e-12)


def raw_rows(b: SupportBench, pool, temp: float) -> pd.DataFrame:
    rows=[]
    for q,vid in enumerate(pool['view_id']):
        sample=b.sample(str(vid)); cm=b.predictive_context_metrics(sample)
        valid=np.flatnonzero((pool['candidate_episode_id'][q]>=0)&np.isfinite(pool['retrieval_score'][q]))
        scores=pool['retrieval_score'][q,valid] if len(valid) else np.asarray([])
        top1=float(np.max(scores)) if len(scores) else -1.0
        rows.append({
            'view_id':str(vid),'view_group_id':str(sample.view_group_id),'split':str(sample.split),'window_start':int(sample.window_start),
            'family':str(sample.view_spec['family']),'severity':float(sample.view_spec['severity']),'distribution':str(sample.view_spec.get('distribution','id')),
            'predictive_context_loss':float(cm['network_mean_context_loss']),
            'network_mean_context_loss':float(cm['network_mean_context_loss']),
            'legacy_context_loss_quantile':float(cm['context_loss_quantile_value']),
            'support_dependent_fraction':float(cm['support_dependent_fraction']),
            'retrieval_ambiguity':entropy_score(scores,temp),'top1_retrieval_score':top1,'valid_candidate_count':int(len(valid)),
        })
    return pd.DataFrame(rows)


def _context_scaler_for_family(cal: dict, family: str) -> dict:
    scalers=cal.get('context_loss_family_scalers',{})
    if family in scalers:
        return scalers[family]
    # Test should normally contain only Validation-fitted families.  A global
    # fallback makes the artifact robust to a newly introduced family while
    # keeping the fallback parameters explicitly recorded in the calibrator.
    if '__global__' in scalers:
        return scalers['__global__']
    raise KeyError(f'No Validation-fitted context-loss scaler for family={family!r}')


def _normalise_context_loss(df: pd.DataFrame, cal: dict) -> np.ndarray:
    values=np.zeros(len(df),dtype=np.float64)
    for family,idx in df.groupby('family').groups.items():
        sc=_context_scaler_for_family(cal,str(family))
        arr=df.loc[idx,'predictive_context_loss'].to_numpy(float)
        if bool(sc.get('constant',False)):
            norm=np.full(len(arr),0.5,dtype=np.float64)
        else:
            lo=float(sc['q_low_value']); hi=float(sc['q_high_value']); den=max(hi-lo,1e-8)
            norm=np.clip((arr-lo)/den,0.0,1.0)
        values[df.index.get_indexer(idx)]=norm
    return values


def apply(df: pd.DataFrame, cal: dict, cfg: dict) -> pd.DataFrame:
    c=cfg['tracks']['support_challenge']; lo=float(cal['coverage_top1_q_low']); hi=float(cal['coverage_top1_q_high'])
    den=max(hi-lo,1e-8); out=df.copy()
    out['memory_coverage']=np.clip((out.top1_retrieval_score.astype(float)-lo)/den,0.0,1.0)
    out['predictive_context_loss_normalized']=_normalise_context_loss(out,cal)
    w1=float(c['context_loss_weight']); w2=float(c['retrieval_ambiguity_weight']); w3=float(c['memory_coverage_weight']); sw=max(w1+w2+w3,1e-12)
    out['support_opportunity']=(w1*out.predictive_context_loss_normalized+w2*out.retrieval_ambiguity+w3*out.memory_coverage)/sw
    low=float(cal['opportunity_low_threshold']); high=float(cal['opportunity_high_threshold'])
    out['opportunity_group']=np.where(out.support_opportunity>=high,'high',np.where(out.support_opportunity<=low,'low','medium'))
    return out


def fit_context_loss_scalers(df: pd.DataFrame, q_low: float, q_high: float) -> dict:
    def one(arr: np.ndarray) -> dict:
        arr=np.asarray(arr,float); arr=arr[np.isfinite(arr)]
        if len(arr)==0:
            return {'q_low_value':0.0,'q_high_value':1.0,'constant':True,'count':0}
        lo=float(np.quantile(arr,q_low)); hi=float(np.quantile(arr,q_high))
        return {
            'q_low_value':lo,
            'q_high_value':hi,
            'constant':bool((hi-lo)<1e-8),
            'count':int(len(arr)),
        }
    scalers={'__global__':one(df.predictive_context_loss.to_numpy(float))}
    for family,g in df.groupby('family',sort=True):
        scalers[str(family)]=one(g.predictive_context_loss.to_numpy(float))
    return scalers


def family_composition(df: pd.DataFrame) -> dict:
    all_counts=df.family.value_counts().sort_index()
    high=df[df.opportunity_group=='high']
    high_counts=high.family.value_counts().reindex(all_counts.index,fill_value=0).sort_index()
    total_high=max(int(high_counts.sum()),1)
    return {
        'all_counts':{str(k):int(v) for k,v in all_counts.items()},
        'high_counts':{str(k):int(v) for k,v in high_counts.items()},
        'high_composition':{str(k):float(v/total_high) for k,v in high_counts.items()},
        'within_family_high_rate':{str(k):float(high_counts[k]/max(int(all_counts[k]),1)) for k in all_counts.index},
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default=None); ap.add_argument('--subgraph',default='TraffiDent_Net_06'); ap.add_argument('--candidate-pool',required=True)
    g=ap.add_mutually_exclusive_group(required=True); g.add_argument('--fit-calibrator',action='store_true'); g.add_argument('--calibrator',default=None)
    ap.add_argument('--output',default=None); ap.add_argument('--challenge-output',default=None); args=ap.parse_args()
    cfg=load_config(args.config); b=SupportBench(args.subgraph,config=args.config); pool=np.load(args.candidate_pool,allow_pickle=False); c=cfg['tracks']['support_challenge']
    df=raw_rows(b,pool,float(c['softmax_temperature']))
    if args.fit_calibrator:
        qlo=float(c['coverage_q_low']); qhi=float(c['coverage_q_high']); top=df.top1_retrieval_score.to_numpy(float)
        lo=float(np.quantile(top,qlo)); hi=float(np.quantile(top,qhi))
        cqlo=float(c.get('context_q_low',0.10)); cqhi=float(c.get('context_q_high',0.90))
        context_scalers=fit_context_loss_scalers(df,cqlo,cqhi)
        provisional={
            'coverage_top1_q_low':lo,
            'coverage_top1_q_high':hi,
            'coverage_quantiles':[qlo,qhi],
            'context_loss_quantiles':[cqlo,cqhi],
            'context_loss_family_scalers':context_scalers,
        }
        tmp=apply(df,{**provisional,'opportunity_low_threshold':0.0,'opportunity_high_threshold':1.0},cfg)
        low=float(np.quantile(tmp.support_opportunity,float(c['low_quantile']))); high=float(np.quantile(tmp.support_opportunity,float(c['high_quantile'])))
        cal={**provisional,'opportunity_low_threshold':low,'opportunity_high_threshold':high,'low_quantile':float(c['low_quantile']),'high_quantile':float(c['high_quantile']),'fitted_split':str(df.split.iloc[0]) if len(df) else 'unknown','future_free':True}
        cal_path=Path(args.output).with_suffix('.calibrator.json') if args.output else Path(args.candidate_pool).with_name(Path(args.candidate_pool).stem+'__support_opportunity_calibrator.json')
        write_json(cal_path,cal); log(f'wrote Validation calibrator {cal_path}')
    else:
        cal=json.loads(Path(args.calibrator).read_text())
    outdf=apply(df,cal,cfg)
    out=Path(args.output) if args.output else Path(args.candidate_pool).with_name(Path(args.candidate_pool).stem+'__support_opportunity.csv'); ensure_dir(out.parent); outdf.to_csv(out,index=False)
    ch=Path(args.challenge_output) if args.challenge_output else out.with_name(out.stem+'__high.csv'); outdf[outdf.opportunity_group=='high'].to_csv(ch,index=False)
    comp=family_composition(outdf)
    comp_path=out.with_name(out.stem+'__family_composition.json')
    write_json(comp_path,comp)
    max_family=max(comp['high_composition'].values(),default=0.0)
    warn=float(c.get('max_high_family_fraction_warning',0.50))
    if max_family>warn:
        log(f'WARNING High-SO family composition is concentrated: max_family_fraction={max_family:.3f} > {warn:.3f}')
    log(f'wrote {out}; low/medium/high={outdf.opportunity_group.value_counts().to_dict()}; challenge={ch}; family_composition={comp_path}')
if __name__=='__main__': main()
