#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline')
if not ROOT.exists(): ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baseline.llm.leaf_qwen import CHOICE_META, build_choice_tensor, build_selector_prompt, multi_ranking_loss, parse_choice, selected_positive_and_negatives

assert len(CHOICE_META)==12
assert CHOICE_META[0]==('GraphBranch','none') and CHOICE_META[6]==('HypergraphBranch','none')
g=torch.arange(12,dtype=torch.float32).view(1,12,1,1).repeat(1,1,3,1)+10
h=g+20
c=build_choice_tensor(g,h)
assert c.shape==(12,12,3,1)
assert torch.allclose(c[0],g[0]) and torch.allclose(c[6],h[0])
sel=np.array([1,7,12],np.uint8)
pos,neg=selected_positive_and_negatives(c,sel,mean=0,std=1)
assert pos.shape==(1,12,3,1) and len(neg)==11
loss=multi_ranking_loss(g,pos,neg)
assert torch.isfinite(loss)
prompt=build_selector_prompt(node_idx=1,num_nodes=3,graph_degree=2,history_raw=np.arange(12),history_observed=np.array([1]*8+[0]*4),history_tod=np.arange(12),history_dow=np.zeros(12),future_tod=np.arange(12,24),future_dow=np.zeros(12),choices_raw=c[...,1,0].numpy())
low=prompt.lower()
for forbidden in ('ground_truth','ground truth','optimal_idx','oracle_choice','future_label','graph_error','hyper_error'):
    assert forbidden not in low
assert 'Choice 12:' in prompt and "?" in prompt
assert parse_choice('2')[0]==2 and parse_choice('Choice 12')[0]==12
assert parse_choice('unknown',fallback=1)==(1,True)
print('LEAF QWEN1.5B CONTRACT SMOKE PASS')
