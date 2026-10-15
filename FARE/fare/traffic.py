"""Directed sparse propagation support, with explicit topology/provenance semantics.

Adjacency is used for context/regions only; directional_neighbors defines directed
candidate propagation paths.  Edge weights are nonnegative strengths, NOT distances.
No shock-wave speed or physical travel time is invented.
"""
from __future__ import annotations
from collections import deque
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class TrafficGraph(nn.Module):
    def __init__(self, adj, directed=None, region_size=24, max_neighbors=4,
                 max_hops=3, path_sources=24):
        super().__init__()
        orig=torch.as_tensor(adj).float()
        if orig.ndim!=2 or orig.shape[0]!=orig.shape[1]:raise ValueError('Adjacency must be square')
        n=orig.shape[0];self.n=n
        a=orig.abs()>0;a.fill_diagonal_(False)
        self.metadata_source='directional_neighbors' if directed is not None else 'adjacency_topology_proxy'
        # Canonical impact/source convention: paths[target,slot] contain source node IDs.
        # up[i,j] is a source upstream of node i; down[i,j] is downstream.
        sources=[[] for _ in range(n)]
        contexts=[set(torch.where(a[i]|a[:,i])[0].tolist()) for i in range(n)]
        directed_edge_count=0
        if directed is not None:
            for target in range(n):
                for name,flag in [('upstream',0),('downstream',1)]:
                    ids=torch.as_tensor(directed[f'{name}_ids']).long()[target]
                    ws=torch.as_tensor(directed[f'{name}_weights']).float()[target]
                    for src,w in zip(ids.tolist(),ws.tolist()):
                        if src<0:continue
                        if src>=n:raise ValueError(f'Out of range {name} node {src}')
                        if src==target or not np.isfinite(w) or w<0:continue
                        # source->target physical-edge affinity, not measured travel time
                        c1=bool(a[src,target]) if name=='upstream' else bool(a[target,src])
                        c2=bool(a[target,src]) if name=='upstream' else bool(a[src,target])
                        confidence=1.0 if c1 else (0.6 if c2 else 0.3)
                        sources[target].append((src,flag,max(float(w),1e-6),confidence))
                        directed_edge_count+=1
        else:
            for i in range(n):
                sources[i]=[(int(j),2,1.0,0.25) for j in contexts[i]]
        self.directed_edges=directed_edge_count
        # BFS across directed edges with bounded branching, preserving direction category.
        # The same path type (upstream/downstream) must be preserved across hops.
        L=min(int(path_sources),n);H=int(max_hops)
        path_nodes=np.zeros((n,L),dtype=np.int64)
        path_type=np.full((n,L),2,dtype=np.int64)
        path_hops=np.zeros((n,L),dtype=np.float32)
        path_weight=np.zeros((n,L),dtype=np.float32)
        path_conf=np.zeros((n,L),dtype=np.float32)
        path_valid=np.zeros((n,L),dtype=bool)
        for target in range(n):
            queue=deque((src,typ,1,w,co) for src,typ,w,co in sources[target])
            best={}
            while queue:
                src,typ,hops,weight,conf=queue.popleft()
                if src==target or hops>H:continue
                key=(src,typ)
                quality=float(weight*conf/(hops**.5))
                if quality<=best.get(key,(-1,))[0]:continue
                best[key]=(quality,hops,weight,conf)
                if hops<H:
                    for nxt,t2,w2,c2 in sources[src]:
                        if t2==typ or typ==2 or t2==2:
                            queue.append((nxt,typ,hops+1,weight*max(w2,1e-4),conf*c2))
            ranked=sorted(((s,t,*info[1:]) for (s,t),info in best.items()),
                          key=lambda x:(-best[(x[0],x[1])][0],x[2],x[0]))[:L]
            for z,(src,typ,hops,w,co) in enumerate(ranked):
                path_nodes[target,z]=src;path_type[target,z]=typ
                path_hops[target,z]=hops;path_weight[target,z]=w
                path_conf[target,z]=co;path_valid[target,z]=True
        for name,value in [('path_nodes',path_nodes),('path_type',path_type),
                           ('path_hops',path_hops),('path_weight',path_weight),
                           ('path_confidence',path_conf),('path_valid',path_valid)]:
            self.register_buffer(name,torch.as_tensor(value))
        # BFS region partition from the broad context graph, not directional paths.
        seen=set();owner=torch.empty(n,dtype=torch.long);region=0
        for start in range(n):
            if start in seen:continue
            queue=deque([start]);pending={start};nodes=[]
            while queue and len(nodes)<region_size:
                i=queue.popleft()
                if i in seen:continue
                seen.add(i);nodes.append(i)
                for j in sorted(contexts[i]):
                    if j not in seen and j not in pending:
                        queue.append(j);pending.add(j)
            owner[nodes]=region;region+=1
        # Region partition never drops disconnected queued nodes.
        if (owner<0).any():raise ValueError('Bad region partition')
        self.g=region;self.register_buffer('owner',owner)
        matrix=torch.zeros(region,n)
        matrix[owner,torch.arange(n)]=1
        self.register_buffer('region_membership',matrix/matrix.sum(-1,keepdim=True).clamp_min(1))
        self.component_neighbors=[sorted(x) for x in contexts]
        self.register_buffer('topology_adj',(a|a.T).float())
        # Compatibility with existing diagnostic and reference implementations.
        self.register_buffer('neighbor_ids',self.path_nodes[:,:max_neighbors].clone())
        self.register_buffer('neighbor_weight',self.path_valid[:,:max_neighbors].float() /
                             self.path_valid[:,:max_neighbors].sum(-1,keepdim=True).clamp_min(1))

    def failure_components(self,mask):
        """Exact components, intended for precompute/diagnostics; NOT inside training forward."""
        b,h,n=mask.shape;absent=(~mask.any(1)).detach().cpu().numpy()
        extent=np.zeros((b,n),dtype=np.float32);boundary=np.zeros((b,n),dtype=np.float32)
        for row in range(b):
            seen=np.zeros(n,dtype=bool)
            for seed in np.flatnonzero(absent[row]):
                if seen[seed]:continue
                stack=[int(seed)];seen[seed]=True;nodes=[];edge=set()
                while stack:
                    src=stack.pop();nodes.append(src)
                    for dst in self.component_neighbors[src]:
                        if absent[row,dst]:
                            if not seen[dst]:seen[dst]=True;stack.append(dst)
                        else:edge.add(dst)
                extent[row,nodes]=len(nodes)/n
                boundary[row,nodes]=min(len(edge)/max(len(nodes),1),1.)
        return torch.from_numpy(extent).to(mask.device),torch.from_numpy(boundary).to(mask.device)

    def save_index(self,path):
        """Save portable static index with verifiable source semantics."""
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
        names=('path_nodes','path_type','path_hops','path_weight','path_confidence','path_valid','owner')
        np.savez_compressed(p,**{k:getattr(self,k).cpu().numpy() for k in names})
        meta={'nodes':self.n,'regions':self.g,'metadata_source':self.metadata_source,
              'directional_entries':self.directed_edges,'path_sources':self.path_nodes.shape[1],
              'physical_distance_available':False,'weight_semantics':'normalized_neighbor_affinity_NOT_distance'}
        p.with_suffix('.json').write_text(json.dumps(meta,indent=2))


class PropagationSupportEncoder(nn.Module):
    def __init__(self,d_base=64,d_hidden=64,horizon=12,regimes=4):
        super().__init__();self.d=d_hidden;self.p=horizon
        self.regime=nn.Sequential(nn.Linear(d_base+6,d_hidden),nn.GELU(),nn.Linear(d_hidden,regimes))
        self.time_lag=nn.Sequential(nn.Linear(d_base+5,d_hidden),nn.GELU(),nn.Linear(d_hidden,2))
        self.embed=nn.Sequential(nn.Linear(d_base+12+regimes,d_hidden),nn.GELU(),nn.Linear(d_hidden,d_hidden))
        self.horizon_embedding=nn.Embedding(horizon,d_hidden)
        self.demand_head=nn.Sequential(nn.Linear(d_hidden+4,d_hidden),nn.GELU(),nn.Linear(d_hidden,1))

    def forward(self,x,mask,hidden,pred,graph,neutral=False,no_direction=False,no_regime=False,
                no_propagation_time=False,no_boundary=False,topology=None):
        b,h,n=x.shape;p=pred.shape[1];dev=x.device
        obs=mask.float();coverage=obs.mean(1)
        mean=(x*obs).sum(1)/obs.sum(1).clamp_min(1)
        trend=x[:,-1]-x[:,0]
        local=torch.stack([mean,trend,coverage,1-coverage,x[:,-1],
                           (pred[:,-1]-pred[:,0])],-1)
        probs=self.regime(torch.cat([hidden,local],-1)).softmax(-1)
        if no_regime:probs=torch.full_like(probs,1/probs.shape[-1])
        # [B,N,L,H] source observability gathered using fixed sparse paths.
        ids=graph.path_nodes;L=ids.shape[1]
        source=obs.permute(0,2,1)[:,ids]       # [B,N,L,H]
        valid=graph.path_valid[None,:,:,None]
        src_weight=graph.path_weight.clamp_min(1e-6)*graph.path_confidence
        src_weight=src_weight*graph.path_valid
        if no_direction:src_weight=graph.path_valid.float()
        # Predicted travel/propagation time is dimensionless in 5-minute intervals.
        temporal_in=torch.cat([hidden,mean[...,None],trend[...,None],coverage[...,None],
                               probs.max(-1).values[...,None],(pred[:,-1]-pred[:,0])[...,None]],-1)
        lag_data=self.time_lag(temporal_in)
        # bounded, nonnegative mean lag; learned uncertainty, no physical speed claim.
        lag_mean=F.softplus(lag_data[...,0])+1.
        lag_sigma=F.softplus(lag_data[...,1])+1.
        lead=torch.arange(p,device=dev,dtype=x.dtype)[None,None,:,None,None]+1
        hist_age=torch.arange(h-1,-1,-1,device=dev,dtype=x.dtype)[None,None,None,None,:]
        hop=graph.path_hops[None,:,:,None].to(x.dtype)
        mean_time=(lag_mean[:,:,None,None]*hop).permute(0,1,3,2)[...,None]
        sigma_time=(lag_sigma[:,:,None,None]*(hop+1).sqrt()).permute(0,1,3,2)[...,None]
        if no_propagation_time:
            time_kernel=torch.ones((b,n,p,L,h),device=dev,dtype=x.dtype)
        else:
            time_kernel=torch.exp(-.5*((lead+hist_age-mean_time)/sigma_time.clamp_min(.25)).square())
        effective=src_weight[None,:,None,:,None]*time_kernel*valid[:, :, None, :, :]
        evidence=(effective*source[:,:,None,:,:]).sum((-1,-2))
        denominator=effective.sum((-1,-2)).clamp_min(1e-7)
        reach=(evidence/denominator).clamp(0,1)       # [B,N,P]
        is_up=graph.path_type.eq(0)[None,:,None,:,None]
        is_down=graph.path_type.eq(1)[None,:,None,:,None]
        def coverage_dir(which):
            ww=effective*which
            return ((ww*source[:,:,None,:,:]).sum((-1,-2))/ww.sum((-1,-2)).clamp_min(1e-7)).clamp(0,1)
        fw=coverage_dir(is_up);bw=coverage_dir(is_down)
        if no_direction:fw=reach;bw=reach
        # boundary from observed fraction in direction-specific 1-hop sources.
        one=graph.path_hops.eq(1)[None,:,None,:,None]
        up_boundary=coverage_dir(is_up & one)
        down_boundary=coverage_dir(is_down & one)
        if no_boundary:up_boundary=up_boundary*0;down_boundary=down_boundary*0
        if topology is None:
            # GPU-only connected-hole proxy. Exact components available offline.
            missing_node=(~mask.any(1)).float()
            hole_extent=((missing_node[:,:,None]*graph.topology_adj[None,:,:]).sum(-1) /
                         graph.topology_adj.sum(-1)[None,:].clamp_min(1))
            hole_boundary=(1-hole_extent)*missing_node
        else:
            hole_extent=topology['component_size'];hole_boundary=topology['boundary_ratio']
        extended=torch.stack([fw,bw,1-fw,1-bw,up_boundary,down_boundary,
                              reach,1-reach,coverage[:,:,None].expand(-1,-1,p),
                              hole_extent[:,:,None].expand(-1,-1,p),
                              hole_boundary[:,:,None].expand(-1,-1,p),
                              pred.permute(0,2,1)],-1)
        # use 12 scalar features after removing redundant (1-reach):
        extended=extended[...,:12]
        hz=self.horizon_embedding(torch.arange(p,device=dev))[None,None,:,:]
        regime=probs[:,:,None,:].expand(-1,-1,p,-1)
        emb=self.embed(torch.cat([hidden[:,:,None,:].expand(-1,-1,p,-1),extended,regime],-1)) + hz
        demand=self.demand_head(torch.cat([emb,1-fw[...,None],1-bw[...,None],
                                           (1-reach)[...,None],(1-coverage[:,:,None]).expand(-1,-1,p)[...,None]],-1)).sigmoid().squeeze(-1)
        if neutral:
            emb=emb*0;demand=demand*0;fw=fw*0;bw=bw*0
        return {'embedding':emb,'demand':demand,'forward_coverage':fw,
                'backward_coverage':bw,'boundary_up':up_boundary.mean(-1),
                'boundary_down':down_boundary.mean(-1),'regime':probs,
                'missingness':1-coverage,'component_size':hole_extent,
                'missing_boundary_ratio':hole_boundary,
                'lag_mean':lag_mean,'lag_sigma':lag_sigma,
                'direction_valid':graph.metadata_source=='directional_neighbors'}
