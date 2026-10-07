from __future__ import annotations
import torch
from torch import nn
from .stla import missing_age

class StateEncoder(nn.Module):
    def __init__(self,num_nodes,history,temporal_dim=64,spatial_dim=64,query_dim=128,retrieval_dim=128,spatial_heads=4,dropout=0.1):
        super().__init__();self.N=num_nodes;self.H=history;self.temporal_dim=temporal_dim;self.spatial_dim=spatial_dim
        self.temporal=nn.Sequential(nn.Linear(history*4,128),nn.GELU(),nn.Dropout(dropout),nn.Linear(128,temporal_dim))
        self.node_emb=nn.Parameter(torch.randn(num_nodes,spatial_dim)*0.02);self.summary=nn.Sequential(nn.Linear(5,spatial_dim),nn.GELU(),nn.Linear(spatial_dim,spatial_dim))
        self.spatial_fuse=nn.Sequential(nn.Linear(spatial_dim*2,spatial_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(spatial_dim,spatial_dim))
        self.node_query=nn.Sequential(nn.Linear(temporal_dim+spatial_dim,query_dim),nn.GELU(),nn.Linear(query_dim,query_dim)); self.temporal_key=nn.Linear(temporal_dim,retrieval_dim);self.spatial_key=nn.Linear(spatial_dim,retrieval_dim)
    def forward(self,x,mask,tod,dow):
        B,H,N=x.shape;age=missing_age(mask);tow=(dow.float()*288.0+tod.float())/2016.0;tow=tow[:,:,None].expand(B,H,N)
        tf=torch.stack([x,mask,age,tow],-1).permute(0,2,1,3).reshape(B,N,H*4);tn=self.temporal(tf)
        den=mask.sum(1).clamp_min(1.0);mean=(x*mask).sum(1)/den;var=(((x-mean[:,None,:])**2)*mask).sum(1)/den;std=torch.sqrt(var.clamp_min(1e-8));cov=mask.mean(1)
        rev=torch.argmax(mask.flip(1),dim=1);idx=(H-1-rev).clamp_min(0);last=torch.gather(x,1,idx[:,None,:]).squeeze(1);last=torch.where(cov>0,last,torch.zeros_like(last));last_age=age[:,-1]
        sf=torch.stack([mean,std,last,cov,last_age],-1);local=self.summary(sf)+self.node_emb[None,:,:];global_ctx=local.mean(1,keepdim=True).expand_as(local);sn=local+self.spatial_fuse(torch.cat([local,global_ctx],-1))
        nq=self.node_query(torch.cat([tn,sn],-1));tk=torch.nn.functional.normalize(self.temporal_key(tn.mean(1)),dim=-1);sk=torch.nn.functional.normalize(self.spatial_key(sn.mean(1)),dim=-1)
        return {'temporal_node':tn,'spatial_node':sn,'node_query':nq,'temporal_key':tk,'spatial_key':sk}
