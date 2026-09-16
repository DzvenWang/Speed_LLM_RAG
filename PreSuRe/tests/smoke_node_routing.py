#!/usr/bin/env python3
from pathlib import Path
import sys,numpy as np,torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from presure.features import row_normalize_adjacency,build_node_features_batch,node_feature_dim,NODE_SENSE_DIM,node_sensing_from_features
from presure.models import NodeM1Model,NodeUsefulnessNet,PairwiseSupportRouter,SensingAwareSafeGate
B,H,P,N,K,L=3,5,4,10,6,3; rng=np.random.default_rng(4)
x=rng.normal(size=(B,H,N)).astype('f4');obs=rng.random((B,H,N))>.25;x[~obs]=0;base=(40+rng.normal(size=(B,P,N))*5).astype('f4');tod=np.tile(np.arange(H),(B,1));dow=np.zeros((B,H));adj=rng.random((N,N)).astype('f4');adj=(adj+adj.T)/2
A=row_normalize_adjacency(adj);f=build_node_features_batch(x,obs,base,tod,dow,A,40,10);assert f.shape==(B,N,node_feature_dim(H,P));s=node_sensing_from_features(f);assert s.shape==(B,N,NODE_SENSE_DIM)
m1=NodeM1Model(f.shape[-1],N,P);corr,z=m1(torch.tensor(f));assert corr.shape==(B,P,N) and z.shape[:2]==(B,N)
c=z[:,None].expand(-1,K,-1,-1)+torch.randn(B,K,N,z.shape[-1])*.01;cs=torch.tensor(s)[:,None].expand(-1,K,-1,-1);sim=torch.randn(B,K);m3=NodeUsefulnessNet(z.shape[-1]);logits,score,mscale=m3(z,c,sim,torch.tensor(s),cs);assert logits.shape==(B,K,N,3)
q=z;csel=c[:,:L].permute(0,2,1,3);cssel=cs[:,:L].permute(0,2,1,3);rr=torch.randn(B,N,L,P)*.2;conf=torch.rand(B,N,L);qctx=torch.tensor(f);cctx=torch.tensor(f)[:,:,None,:].expand(-1,-1,L,-1).contiguous()
r=PairwiseSupportRouter(z.shape[-1],f.shape[-1],P,context_dim=24,residual_dim=12,pair_dim=32,hidden=64,horizon_dim=8,attention_heads=4)
start=4*H; sb=cctx[...,start:start+P];sl,scale,no=r(q,csel,qctx,cctx,torch.tensor(s),cssel,rr,(torch.tensor(base).permute(0,2,1)-40)/10,sb,conf);assert sl.shape==(B,N,L,P) and scale.shape==sl.shape and no.shape==(B,N,P)
w=torch.softmax(torch.cat([sl,no[:,:,None,:]],2),2);ws=w[:,:,:L];cc=(ws*(scale*rr)).sum(2).permute(0,2,1);mass=(1-w[:,:,-1]).permute(0,2,1);dis=torch.zeros_like(cc);g=SensingAwareSafeGate(z.shape[-1],P,N);gg=g(z,torch.tensor(s),(torch.tensor(base)-40)/10,cc/10,mass,dis);assert gg.shape==(B,P,N) and torch.all((gg>=0)&(gg<=1))
print('PRESURE NODE ROUTING V0.7 SMOKE PASS',{'node_features':f.shape,'m3_logits':tuple(logits.shape),'pairwise_routes':tuple(sl.shape),'gate':tuple(gg.shape)})
