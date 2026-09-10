#!/usr/bin/env python3
"""Create evaluator-only outcome labels keyed to explanation input records."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from supportbench.config import load_config
from supportbench.utils import ensure_dir, log


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default=None); ap.add_argument('--utility',required=True); ap.add_argument('--max-pairs',type=int,default=0); ap.add_argument('--gain-key',default=None); ap.add_argument('--output',default=None)
    args=ap.parse_args(); cfg=load_config(args.config); u=np.load(args.utility,allow_pickle=False); gain_key=args.gain_key or str(cfg['support_utility'].get('primary_gain_key','gain_support_dependent'))
    if gain_key not in u.files: gain_key='forecast_gain'
    limit=int(args.max_pairs or cfg['explanation']['evidence_pairs_per_split']); out=Path(args.output) if args.output else Path(args.utility).with_name(Path(args.utility).stem+'__explanation_labels.jsonl'); ensure_dir(out.parent)
    n=0
    with out.open('w',encoding='utf-8') as f:
        for q,vid0 in enumerate(u['view_id']):
            vid=str(vid0)
            for k,eid0 in enumerate(u['candidate_episode_id'][q]):
                eid=int(eid0)
                if eid<0 or not np.isfinite(u[gain_key][q,k]): continue
                rec={'record_id':f'{vid}__e{eid}','view_id':vid,'candidate_episode_id':eid,'primary_gain_key':gain_key,'primary_gain':float(u[gain_key][q,k]),'beneficial':bool(u[gain_key][q,k]>0)}
                for key in ['gain_all','gain_masked','gain_support_dependent','gain_h3','gain_h6','gain_h12']:
                    if key in u.files and np.isfinite(u[key][q,k]): rec[key]=float(u[key][q,k])
                if 'safe_support' in u.files: rec['safe_support']=bool(u['safe_support'][q,k])
                f.write(json.dumps(rec,ensure_ascii=False,allow_nan=False)+'\n'); n+=1
                if n>=limit: break
            if n>=limit: break
    log(f'wrote {out}; evaluator-only records={n}; gain_key={gain_key}')
if __name__=='__main__': main()
