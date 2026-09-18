#!/usr/bin/env python3
from pathlib import Path
import sys,numpy as np,torch,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from presure.query_error_state import QueryErrorStateNet,build_physical_transition_features,build_recent_causal_error_features,RECENT_ERROR_DIM,PHYSICAL_TRANSITION_DIM
from presure.query_error_direction import derive_abs_error_thresholds,query_direction_loss
from presure.backbone_hidden import canonical_node_tensor, discover_hidden_layer

B,N,H,P,F=3,7,12,12,4*12+12+8+4+6
m=QueryErrorStateNet(F,N,P,4*H,hidden_dim=16,recent_dim=RECENT_ERROR_DIM,physical_dim=PHYSICAL_TRANSITION_DIM,use_hidden=True,use_recent=True,use_physical=True,encoder_hidden=32,state_dim=16,node_dim=8,horizon_dim=6,extra_dim=8,hidden=24,dropout=0.0)
x=torch.randn(B,N,F);hid=torch.randn(B,N,16);rec=torch.randn(B,N,RECENT_ERROR_DIM);phy=torch.randn(B,N,PHYSICAL_TRANSITION_DIM)
e,l=m(x,hid,rec,phy);assert e.shape==(B,P,N) and l.shape==(B,P,N)
raw=np.random.default_rng(1).normal(size=(8,P,N)).astype(np.float32);v=np.ones_like(raw,bool);th=derive_abs_error_thresholds(raw,v,(.5,.75,.9));q50=torch.tensor(np.asarray(th['q50'],np.float32)[None,:,None]);loss,_=query_direction_loss(e,l,torch.tensor(raw[:B]),torch.ones_like(e,dtype=torch.bool),1.0,q50,torch.tensor(1.0));loss.backward();assert torch.isfinite(loss)
# Node-axis canonicalization preserves B,N.
z=canonical_node_tensor(torch.randn(B,4,N,8),B,N);assert z.shape==(B,N,32)

# Hidden discovery is auditable even when a module is invoked more than once.
class Reused(torch.nn.Module):
    def __init__(self): super().__init__(); self.act=torch.nn.GELU()
    def forward(self,x): return self.act(x)+self.act(x*0.5)
class A:
    def __init__(self): self.model=Reused()
    def eval_forward(self,b): return self.model(b['x'])
choice=discover_hidden_layer(A(),{'x':torch.randn(B,N,20)},N,{'min_raw_width':16,'max_raw_width':128,'include_regex':'.*','exclude_regex':'^$'})
assert choice.module_name=='act' and choice.call_ordinal in {1,2} and choice.raw_width==20

# Physical features are query-history only and shape-stable.
xx=np.random.default_rng(2).normal(size=(B,H,N)).astype(np.float32);obs=np.ones((B,H,N),bool);adj=np.eye(N,dtype=np.float32)+np.roll(np.eye(N,dtype=np.float32),1,axis=1);p=build_physical_transition_features(xx,obs,adj);assert p.shape==(B,N,PHYSICAL_TRANSITION_DIM)
# Causal recent-error synthetic cache: row anchors advance by 12 five-minute steps.
class DS: pass
d=DS();M=6;d.y=np.random.default_rng(3).normal(size=(M,P,N)).astype(np.float32);base=np.zeros_like(d.y);d.valid=np.ones_like(d.y,bool);d.tod=np.array([[(i*12-h-1)%288 for h in range(H)][::-1] for i in range(M)],np.int64);d.dow=np.zeros((M,H),np.int64);d.meta=pd.DataFrame({'query_index':np.arange(M)*12})
r,man=build_recent_causal_error_features(d,base,expected_stride=12,max_gap_steps=48);assert r.shape==(M,N,RECENT_ERROR_DIM) and man['available'] and np.all(r[0]==0) and np.any(r[1:,:,-1]>0)
print('SMOKE QueryErrorStateProbe v0.9.5 PASS')
