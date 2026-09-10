#!/usr/bin/env python3
"""Go/No-Go diagnostics for retrieval and predictive-support reasoning.

Run this before training Qwen.  It reports whether useful support is present in
the candidate pool, whether naive retrieval often selects harmful support, and
how much oracle selection headroom remains.  The default utility is
support-dependent realised gain, with global-safe support reported separately.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from supportbench.utils import write_json


def finite_rank(ids,score,gain,q):
    v=np.flatnonzero((ids[q]>=0)&np.isfinite(score[q])&np.isfinite(gain[q]))
    if not len(v): return v
    return v[np.argsort(-score[q,v])]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--utility',required=True); ap.add_argument('--gain-key',default=None); ap.add_argument('--ks',nargs='*',type=int,default=[8,16,32,64]); ap.add_argument('--output',default=None)
    args=ap.parse_args(); u=np.load(args.utility,allow_pickle=False); ids=u['candidate_episode_id']; score=u['retrieval_score']
    gain_key=args.gain_key or ('gain_support_dependent' if 'gain_support_dependent' in u.files else 'forecast_gain')
    if gain_key not in u.files: raise KeyError(gain_key)
    gain=u[gain_key]; gain_all=u['gain_all'] if 'gain_all' in u.files else u['forecast_gain']; safe=u['safe_support'].astype(bool) if 'safe_support' in u.files else None
    Q,K=ids.shape; ks=sorted(set(k for k in args.ks if k>0 and k<=K));
    if K not in ks: ks.append(K)
    recall={k:[] for k in ks}; safe_recall={k:[] for k in ks}; top_harm=[]; top_gain=[]; oracle_gain=[]; oracle_safe_gain=[]; valid_counts=[]
    top_all_gain=[]
    for q in range(Q):
        r=finite_rank(ids,score,gain,q); valid_counts.append(len(r))
        if not len(r): continue
        top=int(r[0]); top_harm.append(bool(gain[q,top]<0)); top_gain.append(float(gain[q,top])); top_all_gain.append(float(gain_all[q,top]) if np.isfinite(gain_all[q,top]) else np.nan)
        oracle_gain.append(float(np.nanmax(gain[q,r])))
        if safe is not None:
            sg=[int(i) for i in r if safe[q,i] and np.isfinite(gain[q,i])]
            oracle_safe_gain.append(float(np.max([gain[q,i] for i in sg])) if sg else np.nan)
        for k in ks:
            rr=r[:min(k,len(r))]; recall[k].append(bool(np.any(gain[q,rr]>0)))
            if safe is not None: safe_recall[k].append(bool(np.any(safe[q,rr])))
    report={
        'utility':str(Path(args.utility).resolve()),'gain_key':gain_key,'queries':int(Q),'candidate_k':int(K),'mean_valid_candidate_count':float(np.mean(valid_counts)) if valid_counts else 0.0,
        'beneficial_recall_at_k':{str(k):float(np.mean(recall[k])) if recall[k] else None for k in ks},
        'safe_support_recall_at_k':{str(k):float(np.mean(safe_recall[k])) if safe_recall[k] else None for k in ks} if safe is not None else None,
        'retrieval_top1_harmful_rate':float(np.mean(top_harm)) if top_harm else None,
        'mean_retrieval_top1_gain':float(np.nanmean(top_gain)) if top_gain else None,
        'mean_retrieval_top1_global_gain':float(np.nanmean(top_all_gain)) if top_all_gain else None,
        'mean_oracle_gain_within_pool':float(np.nanmean(oracle_gain)) if oracle_gain else None,
        'mean_oracle_safe_gain_within_pool':float(np.nanmean(oracle_safe_gain)) if oracle_safe_gain and np.any(np.isfinite(oracle_safe_gain)) else None,
        'selection_headroom_gain':float(np.nanmean(np.asarray(oracle_gain)-np.asarray(top_gain))) if oracle_gain and top_gain else None,
        'go_no_go':{
            'low_beneficial_recall':'Improve I1/candidate acquisition before LLM training.',
            'large_selection_headroom':'Matched reasoners (MLP/Transformer/Qwen) have measurable candidate-judgement headroom.',
            'high_top1_harmful_rate':'Similarity/multi-view ranking is unsafe enough to justify support-value reasoning/control.',
        },
    }
    print(json.dumps(report,indent=2)); out=Path(args.output) if args.output else Path(args.utility).with_name(Path(args.utility).stem+'__rag_headroom.json'); write_json(out,report)
if __name__=='__main__': main()
