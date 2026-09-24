#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from oracle.metrics import MetricAccumulator,improvement_pct
from oracle.deep import forward_with_deep_states,decode_prehead
from oracle.observation import ObservationAwarePatchSTG,apply_true_coverage,missingness_features,nested_restore_mask

# Metric sanity: a perfect oracle must improve MAE to zero.
y=np.array([[[1.,2.],[3.,4.]]],np.float32); p0=y+1.; p1=y.copy(); valid=np.ones_like(y,bool); direct=np.array([[1,1]],bool); support=np.array([[1,0]],bool)
a=MetricAccumulator(2); b=MetricAccumulator(2); a.update(p0,y,valid,direct,support); b.update(p1,y,valid,direct,support)
sa=a.summary(); sb=b.summary(); assert abs(sa['masked_mae']-1.)<1e-8; assert abs(sb['masked_mae'])<1e-8; assert improvement_pct(sa['masked_mae'],sb['masked_mae'])==100.0

# Restoration algebra sanity: only hidden raw-valid cells are changed.
xo=np.zeros((2,3),np.float32); cn=np.arange(6,dtype=np.float32).reshape(2,3); rv=np.ones((2,3),bool); ho=np.array([[1,0,1],[0,1,1]],bool); rest=rv&~ho; xc=xo.copy(); xc[rest]=cn[rest]
assert np.all(xc[ho]==0); assert np.all(xc[rest]==cn[rest])

# True cell coverage is exact per query, deterministic, and nested.  It changes
# only real hidden cells and updates the observation mask at the same cells.
r=np.array([[[1,1,0],[0,1,1]],[[1,0,1],[1,0,1]]],bool); ids=np.array([7,11])
s25=nested_restore_mask(r,ids,.25,123); s50=nested_restore_mask(r,ids,.50,123); s100=nested_restore_mask(r,ids,1.,123)
assert np.all(s25<=s50); assert np.all(s50<=s100); assert np.array_equal(s100,r)
assert np.array_equal(s25.sum((1,2)),np.array([1,1])); assert np.array_equal(s50.sum((1,2)),np.array([2,2]))
tx0=torch.zeros(2,2,3); txc=torch.arange(12,dtype=torch.float32).reshape(2,2,3); tobs=torch.zeros(2,2,3,dtype=torch.bool)
txr,tmr,tsel=apply_true_coverage(tx0,txc,tobs,torch.from_numpy(r),torch.from_numpy(ids),.5,123)
assert torch.equal(tmr,tsel); assert torch.equal(txr[tsel],txc[tsel]); assert torch.all(txr[~tsel]==0)
mf=missingness_features(tmr); assert mf.shape==(2,2,3,4); assert torch.all((mf>=0)&(mf<=1))

# Non-invasive PatchSTG deep-interface equivalence on a structural mock.
class Block(nn.Module):
    def __init__(self,d): super().__init__(); self.fc=nn.Linear(d,d,bias=False); nn.init.eye_(self.fc.weight)
    def forward(self,x): return self.fc(x)
class MockPatch(nn.Module):
    def __init__(self,N=4,D=8,P=3):
        super().__init__(); self.node_num=N; self.reo_all_idx=[0,1,2,3,0,1]; self.ori_parts_idx=[0,1,2,3]; self.reo_parts_idx=[0,1,2,3]
        self.input_st_fc=nn.Conv2d(3,2,(1,1),(1,1)); self.node_emb=nn.Parameter(torch.zeros(N,2))
        self.time_in_day_emb=nn.Parameter(torch.zeros(1,2)); self.day_in_week_emb=nn.Parameter(torch.zeros(1,2))
        self.spa_encoder=nn.ModuleList([Block(D) for _ in range(5)]); self.regression_conv=nn.Conv2d(D,P,(1,1)); self.emb=nn.Linear(1,D)
    def embedding(self,x,te): return self.emb(x)
    def forward(self,x,te):
        h=self.embedding(x,te)[:,:,self.reo_all_idx,:]
        for block in self.spa_encoder: h=block(h)
        out=h.new_zeros((h.shape[0],h.shape[1],self.node_num,h.shape[-1])); out[:,:,self.ori_parts_idx,:]=h[:,:,self.reo_parts_idx,:]
        return self.regression_conv(out.transpose(2,3).reshape(out.shape[0],-1,out.shape[2],1)).squeeze(-1)

torch.manual_seed(10); m=MockPatch().eval(); x=torch.randn(2,1,4,1); te=torch.zeros(2,1,4,2,dtype=torch.long)
with torch.no_grad():
    po=m(x,te); pd,states=forward_with_deep_states(m,x,te,(2,4)); pr=decode_prehead(m,states['prehead_4d'])
assert torch.allclose(po,pd,atol=1e-7); assert torch.allclose(pd,pr,atol=1e-7)

# The zero-initialized observation branch exactly preserves PatchSTG.  The
# reconstruction head can change without becoming a forecast input.
aware=ObservationAwarePatchSTG(m,history=1,mask_hidden=4).eval(); observed=torch.ones(2,1,4,dtype=torch.bool)
with torch.no_grad():
    pa,state,_=aware(x,te,observed,return_state=True)
assert torch.allclose(po,pa,atol=1e-7); assert state.shape==(2,1,4)
with torch.no_grad(): aware.state_head.bias.fill_(100.); pa2=aware(x,te,observed)
assert torch.allclose(pa,pa2,atol=1e-7)
print('SMOKE PatchSTG Oracle PASS: real-cell coverage + metrics + mask-aware/state-aux interfaces')
