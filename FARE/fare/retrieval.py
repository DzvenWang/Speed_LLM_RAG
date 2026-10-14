"""Failure-conditioned hierarchical retrieval. Keys contain past history only."""
import torch
from torch import nn
import torch.nn.functional as F
from .memory import _episode_key

class HierarchicalRetriever(nn.Module):
    def __init__(self,d,coarse=64,shortlist=16):
        super().__init__();self.coarse=coarse;self.shortlist=shortlist
        self.support_query=nn.Linear(d,4)
        self.support_gain=nn.Parameter(torch.tensor(0.1))
        self.region_bias=nn.Linear(d,4)
    def forward(self,x,mask,support,origin,memory,graph,purge=0):
        b,h,n=x.shape;keys=memory.keys(x.device)
        q=_episode_key(x,mask).squeeze(0) if False else _episode_key(x,mask) # [B,N,4]
        support_node=support['embedding'].mean(2)
        query=q+self.support_gain.tanh()*torch.tanh(self.support_query(support_node))
        region=graph.region_membership
        qg=torch.einsum('gn,bnd->bgd',region,query)
        kg=torch.einsum('gn,mnd->mgd',region,keys)
        # distance-based coarse candidate recall, conditioned on support.
        score=-((qg[:,:,None,:]-kg.permute(1,0,2)[None,:,:,:]).square()).mean(-1)
        lawful=memory.valid_for(origin,x.device,purge=purge)
        score=score.masked_fill(~lawful[:,None,:],-1e7)
        co=min(self.coarse,memory.m)
        sc,ids=score.topk(co,-1)
        vc=lawful.gather(1,ids.reshape(b,-1)).reshape(b,graph.g,co)
        node_ids=ids[:,graph.owner,:]
        node_valid=vc[:,graph.owner,:]
        node_key=keys[node_ids,torch.arange(n,device=x.device)[None,:,None]]
        node_score=-((query[:,:,None,:]-node_key).square()).mean(-1)
        node_score=node_score.masked_fill(~node_valid,-1e7)
        k=min(self.shortlist,co)
        ss,sl=node_score.topk(k,-1)
        short_ids=node_ids.gather(-1,sl)
        short_valid=node_valid.gather(-1,sl)
        return {'ids':short_ids,'valid':short_valid,'scores':ss,'region_ids':ids,
                'region_valid':vc,'region_scores':sc,'node_query':query}
