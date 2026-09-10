#!/usr/bin/env python3
"""Mine contrastive reasoning supervision from real retrieval candidates.

v1.2.2 defaults to support-dependent realised gain while retaining global gain
as a safety signal.  Pair types explicitly expose where naive retrieval cues
conflict with downstream usefulness.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from supportbench.utils import ensure_dir, log


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--utility',required=True); ap.add_argument('--gain-key',default=None); ap.add_argument('--max-pairs-per-query',type=int,default=4); ap.add_argument('--min-gain-gap',type=float,default=1e-3); ap.add_argument('--output',default=None)
    args=ap.parse_args(); u=np.load(args.utility,allow_pickle=False); rows=[]
    gain_key=args.gain_key or ('gain_support_dependent' if 'gain_support_dependent' in u.files else 'forecast_gain')
    if gain_key not in u.files: raise KeyError(f'{gain_key} not in utility file; keys={u.files}')
    has_t='temporal_score' in u.files; has_p='propagation_score' in u.files; has_s='spatial_score' in u.files
    gain_all=u['gain_all'] if 'gain_all' in u.files else u['forecast_gain']
    safe=u['safe_support'].astype(bool) if 'safe_support' in u.files else None
    support_frac=u['support_dependent_fraction'] if 'support_dependent_fraction' in u.files else np.ones(len(u['view_id']),np.float32)
    for q,vid in enumerate(u['view_id']):
        ids=u['candidate_episode_id'][q]; gain=u[gain_key][q]; sim=u['retrieval_score'][q]
        valid=np.flatnonzero((ids>=0)&np.isfinite(gain)&np.isfinite(sim))
        if len(valid)<2: continue
        candidates=[]
        for a in valid:
            for b in valid:
                if a==b or gain[a] <= gain[b] + args.min_gain_gap: continue
                similarity_conflict=bool(sim[a] < sim[b])
                polarity=bool(gain[a] > 0 and gain[b] < 0)
                temporal_prop=False
                if has_t and has_p:
                    # Negative b looks more temporally similar, but positive a
                    # has stronger propagation compatibility and higher utility.
                    temporal_prop=bool(u['temporal_score'][q,b] > u['temporal_score'][q,a] and u['propagation_score'][q,b] < u['propagation_score'][q,a])
                redundant=bool(float(support_frac[q])<0.05 and sim[b]>=sim[a] and abs(float(gain[b]))<=max(args.min_gain_gap,1e-3))
                priority=(8 if polarity else 0)+(4 if temporal_prop else 0)+(2 if similarity_conflict else 0)+(1 if redundant else 0)
                candidates.append((priority,float(gain[a]-gain[b]),int(a),int(b),similarity_conflict,polarity,temporal_prop,redundant))
        candidates.sort(key=lambda x:(-x[0],-x[1]))
        used=set()
        for _,_,a,b,conflict,polarity,tprop,redundant in candidates:
            key=(int(ids[a]),int(ids[b]))
            if key in used: continue
            used.add(key)
            if polarity: pair_type='beneficial_vs_harmful'
            elif tprop: pair_type='temporal_match_propagation_mismatch'
            elif conflict: pair_type='similarity_conflict'
            elif redundant: pair_type='redundant_support'
            else: pair_type='utility_gap'
            row={
                'view_id':str(vid),'gain_key':gain_key,'pair_type':pair_type,'negative_type':pair_type,
                'positive_episode_id':int(ids[a]),'negative_episode_id':int(ids[b]),
                'positive_gain':float(gain[a]),'negative_gain':float(gain[b]),
                'positive_gain_all':float(gain_all[q,a]) if np.isfinite(gain_all[q,a]) else np.nan,
                'negative_gain_all':float(gain_all[q,b]) if np.isfinite(gain_all[q,b]) else np.nan,
                'positive_similarity':float(sim[a]),'negative_similarity':float(sim[b]),
                'similarity_conflict':conflict,'beneficial_vs_harmful':polarity,
                'support_dependent_fraction':float(support_frac[q]),
            }
            if safe is not None:
                row['positive_safe_support']=bool(safe[q,a]); row['negative_safe_support']=bool(safe[q,b])
            if has_t: row.update({'positive_temporal_score':float(u['temporal_score'][q,a]),'negative_temporal_score':float(u['temporal_score'][q,b])})
            if has_s: row.update({'positive_spatial_score':float(u['spatial_score'][q,a]),'negative_spatial_score':float(u['spatial_score'][q,b])})
            if has_p: row.update({'positive_propagation_score':float(u['propagation_score'][q,a]),'negative_propagation_score':float(u['propagation_score'][q,b])})
            rows.append(row)
            if len(used)>=args.max_pairs_per_query: break
    df=pd.DataFrame(rows); out=Path(args.output) if args.output else Path(args.utility).with_name(Path(args.utility).stem+'__reasoning_pairs.csv'); ensure_dir(out.parent); df.to_csv(out,index=False)
    counts=df.pair_type.value_counts().to_dict() if len(df) and 'pair_type' in df else {}
    log(f'wrote {out}; gain_key={gain_key}; pairs={len(df)}; types={counts}')
if __name__=='__main__': main()
