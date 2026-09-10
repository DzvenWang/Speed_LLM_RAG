#!/usr/bin/env python3
"""Materialise structured records for MLP/Transformer/Qwen reasoner training.

The query future is never part of the reasoner input. Outcome labels are kept
under a dedicated ``label`` field because these are training records, not
explanation-generation inputs.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from supportbench.config import load_config
from supportbench.dataset import SupportBench
from supportbench.evidence import build_pair_evidence
from supportbench.utils import ensure_dir, log


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default=None); ap.add_argument('--subgraph',default='TraffiDent_Net_06'); ap.add_argument('--utility',required=True)
    ap.add_argument('--gain-key',default=None); ap.add_argument('--topk',type=int,default=0); ap.add_argument('--max-records',type=int,default=0); ap.add_argument('--output',default=None)
    args=ap.parse_args(); cfg=load_config(args.config); b=SupportBench(args.subgraph,config=args.config); u=np.load(args.utility,allow_pickle=False)
    gain_key=args.gain_key or str(cfg['reasoning'].get('primary_gain_key','gain_support_dependent'))
    if gain_key not in u.files: gain_key='forecast_gain'
    topk=int(args.topk or cfg['retrieval']['reasoner_candidate_k']); max_anchors=int(cfg['reasoning']['anchor_nodes']); max_lag=int(cfg['reasoning']['propagation_max_lag'])
    out=Path(args.output) if args.output else Path(args.utility).with_name(Path(args.utility).stem+'__reasoning_records.jsonl'); ensure_dir(out.parent)
    n=0
    with out.open('w',encoding='utf-8') as f:
        for q,vid0 in enumerate(u['view_id']):
            vid=str(vid0); sample=b.sample(vid); valid=np.flatnonzero((u['candidate_episode_id'][q]>=0)&np.isfinite(u[gain_key][q]))[:topk]
            for k in valid:
                eid=int(u['candidate_episode_id'][q,k]); evidence=build_pair_evidence(b,sample,eid,max_anchors,max_lag)
                label={'primary_gain_key':gain_key,'primary_gain':float(u[gain_key][q,k]),'beneficial':bool(u[gain_key][q,k]>0)}
                for key in ['gain_all','gain_masked','gain_support_dependent','gain_h3','gain_h6','gain_h12']:
                    if key in u.files and np.isfinite(u[key][q,k]): label[key]=float(u[key][q,k])
                if 'safe_support' in u.files: label['safe_support']=bool(u['safe_support'][q,k])
                rec={
                    'query':{'view_id':vid,'view_group_id':str(sample.view_group_id),'split':str(sample.split),'sensing_family':str(sample.view_spec['family']),'nominal_severity':float(sample.view_spec['severity']),'effective_missing_ratio':float(sample.artificial_mask.mean()),'distribution':str(sample.view_spec.get('distribution','id')),'layout_id':str(sample.view_spec.get('layout_id',''))},
                    'candidate':{'episode_id':eid,'retrieval_score':float(u['retrieval_score'][q,k]),**({name:float(u[name][q,k]) for name in ['temporal_score','spatial_score','propagation_score'] if name in u.files})},
                    'evidence':evidence,'label':label,
                }
                f.write(json.dumps(rec,ensure_ascii=False,allow_nan=False)+'\n'); n+=1
                if args.max_records>0 and n>=args.max_records: break
            if args.max_records>0 and n>=args.max_records: break
            if (q+1)%500==0: log(f'reasoning queries {q+1}/{len(u["view_id"])} records={n}')
    log(f'wrote {out}; records={n}; gain_key={gain_key}; anchors={max_anchors}; max_lag={max_lag}')
if __name__=='__main__': main()
