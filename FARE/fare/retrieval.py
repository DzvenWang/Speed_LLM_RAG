"""Causally filtered, multi-route Region recall and trainable Demand–Capability rerank."""
import torch
from torch import nn
import torch.nn.functional as F
from .memory import _episode_key

class HierarchicalRetriever(nn.Module):
    def __init__(self,d,coarse=64,shortlist=16,chunk=4096):
        super().__init__();self.coarse=int(coarse);self.shortlist=int(shortlist);self.chunk=int(chunk)
        self.state_query=nn.Linear(d,4)
        self.cap_query=nn.Linear(d,8)
        self.matcher=nn.Sequential(nn.Linear(d+4+8+4,d),nn.GELU(),nn.Linear(d,1))
        self.gain=nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _topk_chunked(q,keys,lawful,k,chunk):
        # q[B,G,D], keys[M,G,D], lawful[B,M]. No [B,G,M,D] broadcast.
        b,g,d=q.shape;M=keys.shape[0]
        vals=[];indices=[]
        for low in range(0,M,chunk):
            sub=keys[low:low+chunk]
            score=torch.einsum('bgd,mgd->bgm',F.normalize(q,dim=-1),F.normalize(sub,dim=-1))
            score=score.masked_fill(~lawful[:,None,low:low+len(sub)],-1e7)
            v,i=score.topk(min(k,len(sub)),dim=-1)
            vals.append(v);indices.append(i+low)
        cat=torch.cat(vals,-1);idx=torch.cat(indices,-1)
        top=cat.topk(min(k,cat.shape[-1]),dim=-1)
        return top.values,idx.gather(-1,top.indices)

    def forward(self,x,mask,support,origin,memory,graph,purge=0,intervention=None):
        b,h,n=x.shape;device=x.device
        query=_episode_key(x,mask)
        demand=support['embedding'].mean(2)
        state=query+self.gain.tanh()*self.state_query(demand).tanh()
        cap=F.normalize(self.cap_query(demand),dim=-1)
        r=graph.region_membership
        state_r=torch.einsum('gn,bnd->bgd',r,state)
        cap_r=torch.einsum('gn,bnd->bgd',r,cap)
        key_r,cap_key_r=memory.region_keys(device)
        if intervention in ('no_residual_information','no_evolution_information'):
            cap_key_r=cap_key_r.clone()
            drop=(2,3,7) if intervention=='no_residual_information' else (0,1,4,5,6)
            cap_key_r[...,list(drop)]=0
        lawful=memory.valid_for(origin,device,purge=purge)
        if not lawful.any(-1).all():
            # Legitimate no-memory queries may receive placeholder IDs with valid=False.
            pass
        count=min(self.coarse,memory.m)
        if intervention=='state_similarity_only':
            sc,ids=self._topk_chunked(state_r,key_r,lawful,count,self.chunk)
        else:
            a_count=max(1,count//2);c_count=max(1,count-a_count)
            sc1,id1=self._topk_chunked(state_r,key_r,lawful,a_count,self.chunk)
            sc2,id2=self._topk_chunked(cap_r,cap_key_r,lawful,c_count,self.chunk)
            ids=torch.cat([id1,id2],-1)
            sc=torch.cat([sc1,sc2],-1)
            # Duplicate episodes mask, then refill with state candidates if needed.
            for j in range(1,ids.shape[-1]):
                dup=(ids[...,:j]==ids[...,j:j+1]).any(-1)
                sc[...,j]=sc[...,j].masked_fill(dup,-1e7)
        rvalid=lawful.gather(1,ids.reshape(b,-1)).reshape_as(ids) & (sc>-1e6)
        node_ids=ids[:,graph.owner,:]
        node_valid=rvalid[:,graph.owner,:]
        # Compact node keys kept on CPU, gathered ONLY after region shortlist.
        node_key=memory.fetch_node_keys(node_ids,device,'node_key_cpu')
        node_cap=memory.fetch_node_keys(node_ids,device,'capability_cpu')
        if intervention in ('no_residual_information','no_evolution_information'):
            node_cap=node_cap.clone()
            drop=(2,3,7) if intervention=='no_residual_information' else (0,1,4,5,6)
            node_cap[...,list(drop)]=0
        cap_q=cap[:,:,None,:].expand_as(node_cap)
        state_dist=-(state[:,:,None,:]-node_key).square().mean(-1)
        if intervention=='state_similarity_only':
            score=state_dist
        else:
            d=demand[:,:,None,:].expand(-1,-1,node_ids.shape[-1],-1)
            compat=(cap_q*F.normalize(node_cap,dim=-1)).sum(-1)
            feat=torch.cat([d,node_key,node_cap,
                            (state[:,:,None,:]-node_key).abs()],-1)
            score=state_dist+.2*compat+.3*self.matcher(feat).squeeze(-1)
        score=score.masked_fill(~node_valid,-1e7)
        k=min(self.shortlist,ids.shape[-1])
        v,sl=score.topk(k,-1)
        cid=node_ids.gather(-1,sl)
        valid=node_valid.gather(-1,sl)
        # The information contained in an invalid placeholder cannot be consumed.
        return {'ids':cid,'valid':valid,'scores':v,'region_ids':ids,
                'region_valid':rvalid,'region_scores':sc,'node_query':state,
                'candidate_capability':node_cap.gather(2,sl[...,None].expand(-1,-1,-1,8)),
                'state_query':state,'demand_query':cap}
