#!/usr/bin/env python3
"""Create generator-visible grounded explanation inputs only.

Outcome labels are intentionally impossible to access from this file.  Use
``build_explanation_labels.py`` for evaluator-only realised utility labels.
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
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default=None); ap.add_argument('--subgraph',default='TraffiDent_Net_06'); ap.add_argument('--candidate-pool',required=True); ap.add_argument('--max-pairs',type=int,default=0); ap.add_argument('--output',default=None)
    args=ap.parse_args(); cfg=load_config(args.config); b=SupportBench(args.subgraph,config=args.config); pool=np.load(args.candidate_pool,allow_pickle=False)
    max_pairs=int(args.max_pairs or cfg['explanation']['evidence_pairs_per_split']); max_anchors=int(cfg['reasoning']['anchor_nodes']); max_lag=int(cfg['reasoning']['propagation_max_lag'])
    out=Path(args.output) if args.output else Path(args.candidate_pool).with_name(Path(args.candidate_pool).stem+'__explanation_inputs.jsonl'); ensure_dir(out.parent)
    n=0
    with out.open('w',encoding='utf-8') as f:
        for q,vid0 in enumerate(pool['view_id']):
            vid=str(vid0); sample=b.sample(vid)
            for k,eid0 in enumerate(pool['candidate_episode_id'][q]):
                eid=int(eid0)
                if eid<0: continue
                rec={'record_id':f'{vid}__e{eid}','view_id':vid,'candidate_episode_id':eid,
                     'query_context':{'sensing_family':str(sample.view_spec['family']),'nominal_severity':float(sample.view_spec['severity']),'effective_missing_ratio':float(sample.artificial_mask.mean()),'distribution':str(sample.view_spec.get('distribution','id')),'layout_id':str(sample.view_spec.get('layout_id',''))},
                     'retrieval_score':float(pool['retrieval_score'][q,k]),
                     'retrieval_components':{name.replace('_score',''):float(pool[name][q,k]) for name in ['temporal_score','spatial_score','propagation_score'] if name in pool.files},
                     'evidence':build_pair_evidence(b,sample,eid,max_anchors,max_lag)}
                f.write(json.dumps(rec,ensure_ascii=False,allow_nan=False)+'\n'); n+=1
                if n>=max_pairs: break
            if n>=max_pairs: break
    log(f'wrote {out}; generator-visible records={n}; outcome labels physically excluded')
if __name__=='__main__': main()
