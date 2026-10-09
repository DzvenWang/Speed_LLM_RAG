"""Graph-neighbour observation encoder, replacing global-mean spatial key."""
from __future__ import annotations
import torch
from torch import nn
from .stla import missing_age

class StateEncoder(nn.Module):
    def __init__(self,num_nodes,history,temporal_dim=64,spatial_dim=64,query_dim=128,
                 retrieval_dim=128,spatial_heads=4,dropout=0.1,graph_k=8):
        super().__init__()
        self.N=int(num_nodes);self.H=int(history)
        if spatial_dim % int(spatial_heads):
            raise ValueError('spatial_dim must be divisible by spatial_heads')
        self.heads=int(spatial_heads);self.head_dim=spatial_dim//self.heads
        self.temporal=nn.Sequential(nn.Linear(history*4,128),nn.GELU(),nn.Dropout(dropout),nn.Linear(128,temporal_dim))
        self.node_emb=nn.Parameter(torch.randn(num_nodes,spatial_dim)*.02)
        self.summary=nn.Sequential(nn.Linear(5,spatial_dim),nn.GELU(),nn.Linear(spatial_dim,spatial_dim))
        self.graph_q=nn.Linear(spatial_dim,spatial_dim)
        self.graph_k=nn.Linear(spatial_dim,spatial_dim)
        self.graph_v=nn.Linear(spatial_dim,spatial_dim)
        self.spatial_fuse=nn.Sequential(nn.Linear(spatial_dim*2+1,spatial_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(spatial_dim,spatial_dim))
        self.node_query=nn.Sequential(nn.Linear(temporal_dim+spatial_dim,query_dim),nn.GELU(),nn.Linear(query_dim,query_dim))
        self.temporal_key=nn.Linear(temporal_dim,retrieval_dim)
        self.spatial_key=nn.Linear(spatial_dim,retrieval_dim)
        self.register_buffer('neighbors',torch.arange(num_nodes).view(-1,1).expand(-1,min(graph_k,max(1,num_nodes-1))).clone())

    def set_graph(self,neighbors):
        neighbors=torch.as_tensor(neighbors,device=self.neighbors.device,dtype=torch.long)
        if neighbors.shape != self.neighbors.shape:
            raise ValueError(f'Graph shape {tuple(neighbors.shape)} != configured {tuple(self.neighbors.shape)}')
        if neighbors.min()<0 or neighbors.max()>=self.N:raise ValueError('Graph has invalid node ids')
        self.neighbors.copy_(neighbors)

    def forward(self,x,mask,tod,dow):
        B,H,N=x.shape
        age=missing_age(mask);tow=(dow.float()*288+tod.float())/2016.
        time=tow[:,:,None].expand(B,H,N)
        tf=torch.stack([x,mask,age,time],-1).permute(0,2,1,3).reshape(B,N,H*4)
        tn=self.temporal(tf)
        den=mask.sum(1).clamp_min(1.);avg=(x*mask).sum(1)/den
        var=(((x-avg[:,None])**2)*mask).sum(1)/den
        std=var.clamp_min(1e-8).sqrt();cov=mask.mean(1)
        rev=torch.argmax(mask.flip(1),1);ix=(H-1-rev).clamp_min(0)
        last=torch.gather(x,1,ix[:,None]).squeeze(1)
        last=torch.where(cov>0,last,torch.zeros_like(last))
        sf=torch.stack([avg,std,last,cov,age[:,-1]],-1)
        local=self.summary(sf)+self.node_emb[None]
        neighbours=local[:,self.neighbors]  # B,N,K,D
        ncov=cov[:,self.neighbors]
        qry=self.graph_q(local).reshape(B,N,self.heads,self.head_dim)
        key=self.graph_k(neighbours).reshape(B,N,neighbours.shape[2],self.heads,self.head_dim)
        val=self.graph_v(neighbours).reshape(B,N,neighbours.shape[2],self.heads,self.head_dim)
        scores=(qry[:,:,None]*key).sum(-1)/(self.head_dim**0.5)
        scores=scores.masked_fill(ncov[...,None]<=0.,-1e4)
        weight=torch.softmax(scores,dim=2)*(ncov[...,None]>0.).float()
        ctx=(weight[...,None]*val).sum(2).reshape(B,N,-1)
        n_support=ncov.mean(2)
        spatial=local+self.spatial_fuse(torch.cat([local,ctx,n_support[...,None]],-1))
        nq=self.node_query(torch.cat([tn,spatial],-1))
        wt=cov.clamp_min(0.05)
        t_global=(tn*wt[...,None]).sum(1)/wt.sum(1).clamp_min(1e-5)[...,None]
        s_global=(spatial*wt[...,None]).sum(1)/wt.sum(1).clamp_min(1e-5)[...,None]
        tk=torch.nn.functional.normalize(self.temporal_key(t_global),dim=-1)
        sk=torch.nn.functional.normalize(self.spatial_key(s_global),dim=-1)
        return {'temporal_node':tn,'spatial_node':spatial,'node_query':nq,
                'temporal_key':tk,'spatial_key':sk,'node_coverage':cov,'neighbour_coverage':n_support}
