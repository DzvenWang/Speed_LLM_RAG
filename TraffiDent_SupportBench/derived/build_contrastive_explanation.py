#!/usr/bin/env python3
"""Build physically separated contrastive explanation inputs and labels."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from supportbench.config import load_config
from supportbench.dataset import SupportBench
from supportbench.evidence import build_pair_evidence
from supportbench.utils import ensure_dir, log


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default=None); ap.add_argument('--subgraph',default='TraffiDent_Net_06'); ap.add_argument('--reasoning-pairs',required=True); ap.add_argument('--max-pairs',type=int,default=0); ap.add_argument('--output',default=None); ap.add_argument('--labels-output',default=None)
    args=ap.parse_args(); cfg=load_config(args.config); b=SupportBench(args.subgraph,config=args.config); pairs=pd.read_csv(args.reasoning_pairs)
    lim=int(args.max_pairs or cfg['explanation']['contrastive_pairs_per_split']); inp=Path(args.output) if args.output else Path(args.reasoning_pairs).with_name(Path(args.reasoning_pairs).stem+'__contrastive_inputs.jsonl'); lab=Path(args.labels_output) if args.labels_output else inp.with_name(inp.stem.replace('_inputs','')+'__labels.jsonl'); ensure_dir(inp.parent); ensure_dir(lab.parent)
    max_anchors=int(cfg['reasoning']['anchor_nodes']); max_lag=int(cfg['reasoning']['propagation_max_lag']); n=0
    with inp.open('w',encoding='utf-8') as fi, lab.open('w',encoding='utf-8') as fl:
        for _,r in pairs.head(lim).iterrows():
            sample=b.sample(str(r.view_id)); a=int(r.positive_episode_id); bb=int(r.negative_episode_id); rid=f'{r.view_id}__a{a}__b{bb}'
            input_rec={'record_id':rid,'view_id':str(r.view_id),'candidate_a_episode_id':a,'candidate_b_episode_id':bb,'candidate_a_evidence':build_pair_evidence(b,sample,a,max_anchors,max_lag),'candidate_b_evidence':build_pair_evidence(b,sample,bb,max_anchors,max_lag),'task':'Compare the two historical supports using only the supplied verifiable evidence. Do not assume which candidate is better.'}
            label_rec={'record_id':rid,'view_id':str(r.view_id),'preferred_episode_id':a,'rejected_episode_id':bb,'pair_type':str(r.get('pair_type','utility_gap')),'preferred_gain':float(r.positive_gain),'rejected_gain':float(r.negative_gain)}
            fi.write(json.dumps(input_rec,ensure_ascii=False,allow_nan=False)+'\n'); fl.write(json.dumps(label_rec,ensure_ascii=False,allow_nan=False)+'\n'); n+=1
    log(f'wrote {inp} and {lab}; contrastive pairs={n}; outcomes physically separated')
if __name__=='__main__': main()
