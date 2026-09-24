#!/usr/bin/env python3
from pathlib import Path
import sys,numpy as np,torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from presure.oracles import oracle_global_batch,oracle_horizon_batch,oracle_node_horizon_batch,oracle_node_batch,oracle_region_horizon_batch,oracle_node_selected_batch,oracle_node_horizon_selected_batch
B,K,P,N=2,5,3,7
torch.manual_seed(1);base=torch.zeros(B,P,N);y=torch.randn(B,P,N);valid=torch.ones(B,P,N,dtype=torch.bool);direct=torch.tensor([[1,0,1,0,1,0,1],[0,1,0,1,0,1,0]],dtype=torch.bool);support=~direct;rr=torch.randn(B,K,P,N)
w={'overall':0.45,'masked':0.35,'support':0.20};alphas=[0.25,0.5,1.0];regions=np.array([0,0,0,1,1,2,2],np.int32)
A,*_=oracle_global_batch(base,y,valid,direct,support,rr,alphas,w);C,*_=oracle_horizon_batch(base,y,valid,direct,support,rr,alphas,w);D,*_=oracle_node_horizon_batch(base,y,valid,rr,alphas);E,*_=oracle_node_batch(base,y,valid,rr,alphas);F,*_=oracle_region_horizon_batch(base,y,valid,direct,support,rr,alphas,regions,w)
rrsel=rr[:,:3].permute(0,3,1,2).contiguous();cm=torch.ones(B,N,3,dtype=torch.bool);ES,*_=oracle_node_selected_batch(base,y,valid,rrsel,alphas,cm);DS,*_=oracle_node_horizon_selected_batch(base,y,valid,rrsel,alphas,cm)
def mae(x):return torch.abs(x-y).mean().item()
for x in [A,C,D,E,F,ES,DS]:assert np.isfinite(mae(x))
assert mae(D)<=mae(base)+1e-7
print('PRESURE ORACLE MATH SMOKE PASS',{'base':mae(base),'A':mae(A),'C':mae(C),'E':mae(E),'F':mae(F),'D':mae(D)})
