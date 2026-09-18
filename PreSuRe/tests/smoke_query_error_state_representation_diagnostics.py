#!/usr/bin/env python3
from pathlib import Path
import sys,numpy as np,torch,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from presure.backbone_hidden import discover_hidden_layers,select_sweep_candidates,canonical_node_tensor
from presure.query_error_state import build_repaired_causal_error_features,RECENT_ERROR_REPAIRED_DIM

B,N=3,7
# Parent block keeps N axis even when its leaf projection is input-like; all-module
# discovery must see both and preserve call ordinals.
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__();self.input_st_fc=torch.nn.Linear(20,20);self.deep=torch.nn.Sequential(torch.nn.Linear(20,24),torch.nn.GELU(),torch.nn.Linear(24,20))
    def forward(self,x):
        y=self.input_st_fc(x);return y+self.deep(y)
class Model(torch.nn.Module):
    def __init__(self):super().__init__();self.block1=Block();self.block2=Block()
    def forward(self,x):return self.block2(self.block1(x))
class A:
    def __init__(self):self.model=Model()
    def eval_forward(self,b):return self.model(b['x'])
a=A();cat=discover_hidden_layers(a,{'x':torch.randn(B,N,20)},N,{'min_raw_width':16,'max_raw_width':256,'include_regex':'.*','exclude_regex':'(^$|output|head$|predictor)'})
names=[c.module_name for c in cat]
assert 'block1' in names and 'block2' in names and any('input_st_fc' in x for x in names),names
sel=select_sweep_candidates(cat,{'max_candidates':5});assert 1<=len(sel)<=5 and any(c.module_name=='block2' for c in sel)
assert canonical_node_tensor(torch.randn(B,4,N,8),B,N).shape==(B,N,32)

# Repaired chronology: shuffled target rows and duplicate synthetic views should not
# depend on row order. Validation-like targets use the Train pool only.
class DS:pass
P=12;M=8
d=DS();rng=np.random.default_rng(3);d.y=rng.normal(size=(M,P,N)).astype(np.float32);base=np.zeros_like(d.y);d.valid=np.ones_like(d.y,bool)
# Four anchors, two synthetic views per anchor; intentionally shuffled row order.
ws=np.array([24,0,12,0,36,24,12,36],np.int64)
d.meta=pd.DataFrame({'window_start':ws,'view_group_id':['b','a','b','b','a','a','a','b']})
# Unused by repaired manifest chronology but present to match CacheSplit surface.
d.tod=np.zeros((M,12),np.int64);d.dow=np.zeros((M,12),np.int64)
r,man=build_repaired_causal_error_features(d,base,target_ds=d,history=12,max_lag_queries=4,max_gap_steps=100)
assert r.shape==(M,N,RECENT_ERROR_REPAIRED_DIM) and man['available'],man
# Earliest anchor rows have no earlier realised forecast; later anchors do.
earliest=np.where(ws==0)[0];later=np.where(ws>=24)[0]
assert np.all(r[earliest,:,-1]==0) and np.any(r[later,:,-1]>0)
assert man['pool_chronology']['source']=='manifest_window_start_plus_history'
print('SMOKE QueryErrorStateRepresentationDiagnostics v0.9.5.1 PASS')
