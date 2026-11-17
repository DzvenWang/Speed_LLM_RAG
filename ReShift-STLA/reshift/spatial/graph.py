"""Sparse source->target road and optional TRAIN-only CF sensitivity edges.
No use of validation/test targets or historical retrieval memory.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn


def build_road(root,target,dataset,k_per_direction=4):
    d=Path(root)/'data/processed/traffident/base_supportbench'/target/dataset
    nodes=pd.read_csv(d/'core/nodes.csv')
    adjacency=np.load(d/'graph/adjacency.npy')
    distance=np.load(d/'graph/distance.npy')
    n=len(nodes);assert adjacency.shape==(n,n) and distance.shape==(n,n)
    if not np.array_equal(nodes.local_idx.to_numpy(),np.arange(n)):
        raise RuntimeError('Road nodes local_idx must match dataset node order')
    for field in ('lat','lng','road_direction'):
        if field not in nodes:raise RuntimeError(f'Missing road field {field}')
    lat=pd.to_numeric(nodes.lat,errors='coerce').to_numpy(float)
    lon=pd.to_numeric(nodes.lng,errors='coerce').to_numpy(float)
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():raise RuntimeError('Bad graph GPS')
    direction=nodes.road_direction.astype(str).to_numpy()
    progress=np.empty(n,dtype=float)
    for i,r in enumerate(direction):
        way=r.rsplit('__',1)[-1].upper()
        if way=='N':progress[i]=lat[i]
        elif way=='S':progress[i]=-lat[i]
        elif way=='E':progress[i]=lon[i]
        elif way=='W':progress[i]=-lon[i]
        else:raise RuntimeError(f'Unknown road_direction {r} at node {i}')
    latrad=np.deg2rad(lat);lonrad=np.deg2rad(lon)
    a=np.sin((latrad[:,None]-latrad[None,:])/2)**2+np.cos(latrad[:,None])*np.cos(latrad[None,:])*np.sin((lonrad[:,None]-lonrad[None,:])/2)**2
    gps=12742000*np.arcsin(np.sqrt(np.clip(a,0,1)))
    # Undirected candidate adjacency, road progress determines physical direction.
    cand=(np.isfinite(adjacency)&(adjacency!=0));cand=cand|cand.T
    np.fill_diagonal(cand,False)
    k=2*k_per_direction
    ids=np.zeros((n,k),np.int64);valid=np.zeros((n,k),bool);edges=np.zeros((n,k,5),np.float32)
    adjacency_list=[set() for _ in range(n)]
    for i in range(n):
        for side in (0,1):
            same=direction==direction[i]
            ordered=(progress<progress[i]-1e-8) if side==0 else (progress>progress[i]+1e-8)
            c=np.flatnonzero(cand[i]&same&ordered&(gps[i]>0))
            c=c[np.lexsort((c,gps[i,c]))][:k_per_direction]
            for offset,j in enumerate(c):
                pos=side*k_per_direction+offset
                ids[i,pos]=j;valid[i,pos]=True
                dist=float(distance[i,j]);real_dist=dist>0 and np.isfinite(dist)
                edges[i,pos]=[np.log1p(gps[i,j]/1000),float(side==0),float(side==1),
                              np.log1p(max(dist,0)/1000) if real_dist else 0,float(real_dist)]
                adjacency_list[i].add(int(j));adjacency_list[int(j)].add(i)
    return {'ids':ids,'valid':valid,'edge_features':edges,'direction_labels':direction,
            'adjacency_list':[sorted(v) for v in adjacency_list]}


def add_cf_edges(road,cf_path,max_cf_neighbors=8):
    """Merge precomputed TRAIN-only CF scores into road neighbours; append non-road CF sources."""
    if cf_path is None:return road
    with np.load(cf_path,allow_pickle=False) as f:
        cf_ids=f['source_ids'].astype(np.int64)
        cf_valid=f['valid'].astype(bool)
        cf_sensitivity=f['sensitivity'].astype(np.float32)
        cf_gain=f['gain'].astype(np.float32)
    n,k0=road['ids'].shape
    if cf_ids.shape[0]!=n:raise ValueError('CF graph num_nodes mismatch')
    k=max_cf_neighbors
    ids=np.concatenate([road['ids'],np.zeros((n,k),np.int64)],axis=1)
    valid=np.concatenate([road['valid'],np.zeros((n,k),bool)],axis=1)
    features=np.concatenate([road['edge_features'],np.zeros((n,k,5),np.float32)],axis=1)
    # Separate learned features: type0=road, type1=CF
    typ=np.concatenate([np.zeros((n,k0),np.int64),np.ones((n,k),np.int64)],axis=1)
    for i in range(n):
        used=set(ids[i,valid[i]].tolist())
        pos=0
        for j,s,g,good in zip(cf_ids[i],cf_sensitivity[i],cf_gain[i],cf_valid[i]):
            if not good or int(j)==i or int(j) in used:continue
            if pos>=k:break
            ids[i,k0+pos]=j;valid[i,k0+pos]=True
            features[i,k0+pos,0]=float(np.log1p(max(s,0)))
            features[i,k0+pos,3]=float(np.tanh(g))
            used.add(int(j));pos+=1
    return dict(road,ids=ids,valid=valid,edge_features=features,edge_type=typ)


class GraphEvidence(nn.Module):
    """Per-horizon sparse graph message; explicit no-graph logit and reliability."""
    def __init__(self,graph,dim=32):
        super().__init__()
        self.register_buffer('ids',torch.from_numpy(graph['ids']).long())
        self.register_buffer('valid',torch.from_numpy(graph['valid']).bool())
        self.register_buffer('edge',torch.from_numpy(graph['edge_features']).float())
        edge_type=graph.get('edge_type',np.zeros_like(graph['ids']))
        self.register_buffer('edge_type',torch.from_numpy(edge_type).long())
        self.q=nn.Linear(dim,dim);self.k=nn.Linear(dim,dim);self.v=nn.Linear(dim,dim)
        self.edge_score=nn.Sequential(nn.Linear(6,dim//2),nn.GELU(),nn.Linear(dim//2,1))
        self.null_score=nn.Parameter(torch.tensor(-1.0))
        self.out=nn.Linear(dim,dim)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
        self.rel_bias=nn.Linear(1,1,bias=False)
        nn.init.constant_(self.rel_bias.weight,0.5)

    def forward(self,z,obs_reliability):
        # z [B,P,N,D], obs_reliability [B,N]
        B,P,N,D=z.shape
        if N!=self.ids.shape[0] or obs_reliability.shape!=(B,N):raise ValueError('Graph node mismatch')
        query=self.q(z)
        nk=self.k(z[:,:,self.ids,:])       # [B,P,N,K,D]
        nv=self.v(z[:,:,self.ids,:])
        scores=(query.unsqueeze(3)*nk).sum(-1)/(D**0.5)
        edges=torch.cat([self.edge,self.edge_type.unsqueeze(-1)],dim=-1)
        scores=scores+self.edge_score(edges.float()).squeeze(-1)[None,None]
        neigh_rel=obs_reliability[:,self.ids] # [B,N,K]
        scores=scores+self.rel_bias(neigh_rel[...,None]).squeeze(-1)[:,None]
        scores=scores.masked_fill(~self.valid[None,None],-1.e4)
        null=self.null_score.expand(B,P,N,1)
        probs=torch.softmax(torch.cat([scores.float(),null.float()],dim=-1),dim=-1).to(z.dtype)
        message=(probs[...,:-1,None]*nv).sum(3)
        evidence=self.out(message)
        coverage=(probs[...,:-1]*neigh_rel[:,None]).sum(-1)
        return {'tokens':evidence,'confidence':coverage.clamp(0,1),
                'null_weight':probs[...,-1], 'attention':probs}
