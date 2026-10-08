from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

def masked_softmax(logits,valid,dim=-1):
    safe=logits.masked_fill(~valid,-1e4)
    p=torch.softmax(safe,dim=dim)*valid.float()
    return p/p.sum(dim=dim,keepdim=True).clamp_min(1e-9)

class RetrievalEncoder(nn.Module):
    def __init__(self,history,hidden=64,key_dim=128):
        super().__init__();self.q=nn.Sequential(nn.Linear(history*2+2,hidden),nn.GELU(),nn.Linear(hidden,key_dim))
        self.k=nn.Sequential(nn.Linear(history*2+2,hidden),nn.GELU(),nn.Linear(hidden,key_dim))
    def _features(self,x,valid,extra):
        # [B,G,H,A] -> [B,G,2H+2], with validity masking.
        m=valid.float(); v=(x*m).sum(-1)/m.sum(-1).clamp_min(1)
        c=m.mean(-1)
        return torch.cat([v,c,extra],-1)
    def query(self,ctx):
        f=self._features(ctx['anchor_values'],ctx['anchor_mask'],torch.stack([ctx['missing'],ctx['coverage']],-1))
        return F.normalize(self.q(f),dim=-1)
    @torch.no_grad()
    def memory_keys(self,memory,regions,batch=256):
        ids=regions.anchor_ids;v=regions.anchor_valid;chunks=[]
        for lo in range(0,len(memory.x),batch):
            x=memory.x[lo:lo+batch];valid=memory.valid[lo:lo+batch]
            b,h,n=x.shape; xx=x[:,:,ids.flatten()].view(b,h,regions.g,regions.a).permute(0,2,1,3)
            mm=valid[:,:,ids.flatten()].view(b,h,regions.g,regions.a).permute(0,2,1,3)&v[None,:,None,:]
            ctx={'anchor_values':xx,'anchor_mask':mm}
            coverage=mm.float().mean((-1,-2))
            feat=self._features(xx,mm,torch.stack([torch.zeros_like(coverage),coverage],-1))
            chunks.append(F.normalize(self.k(feat),dim=-1))
        return torch.cat(chunks,0)

class TargetRetriever(nn.Module):
    def __init__(self,history,hidden=64,key_dim=128,coarse_topk=32,topk=8,overlap=24):
        super().__init__();self.encoder=RetrievalEncoder(history,hidden,key_dim)
        self.coarse_topk=coarse_topk;self.topk=topk;self.overlap=overlap
        self.rerank=nn.Sequential(nn.Linear(5,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.log_temperature=nn.Parameter(torch.tensor(math.log(.3)))
    @torch.no_grad()
    def refresh(self,memory,regions):
        was_training=self.encoder.training;self.encoder.eval()
        memory.keys=self.encoder.memory_keys(memory,regions)
        self.encoder.train(was_training)
    def forward(self,ctx,regions,memory,window_start,mode='full'):
        if memory.keys is None:raise RuntimeError('Memory index not built; call retriever.refresh')
        b,g,h,a=ctx['anchor_values'].shape
        query=self.encoder.query(ctx)
        # [B,G,M] from region-indexed keys, chunking avoided huge temporary 4D.
        if mode=='global_retrieval':
            global_query=F.normalize(query.mean(1),dim=-1)
            global_mem=F.normalize(memory.keys.detach().mean(1),dim=-1)
            global_score=global_query@global_mem.T
            scores=global_score[:,None,:].expand(b,g,-1)
        else:
            scores=torch.einsum('bgd,mgd->bgm',query,memory.keys.detach())
        # Exclude every overlapping history+forecast window (including same source sample).
        allowable=(window_start[:,None]-memory.starts[None,:]).abs()>=self.overlap
        scores=scores.masked_fill(~allowable[:,None,:],-1e4)
        count=min(self.coarse_topk,len(memory.x));coarse,id64=scores.topk(count,dim=-1)
        if mode=='random_memory':
            # Deterministic candidate control with the same legal overlap filter.
            random_score=torch.frac(torch.sin((torch.arange(len(memory.x),device=scores.device)[None,None,:]+1)*12.9898+(window_start[:,None,None].float()+1)*78.233)*43758.5453)
            random_score=random_score.expand(b,g,-1).masked_fill(~allowable[:,None,:],-1e4)
            coarse,id64=random_score.topk(count,dim=-1)
        xv,mv=memory.fetch(id64)  # [B,G,K,H,N] temporarily, freed after anchor gathering
        # Query and candidate are compared at the identical actually visible positions.
        aid=regions.anchor_ids
        ar=torch.gather(xv, -1,aid[None,:,None,None,:].expand(b,g,count,h,a))
        av=torch.gather(mv, -1,aid[None,:,None,None,:].expand(b,g,count,h,a))
        del xv,mv
        valid=ctx['anchor_mask'][:,:,None] & av
        denom=valid.sum((-1,-2)).float()
        delta=ctx['anchor_values'][:,:,None]-ar
        mae=(F.smooth_l1_loss(delta,torch.zeros_like(delta),reduction='none')*valid).sum((-1,-2))/denom.clamp_min(1)
        # Trend comparison uses only consecutive commonly observed positions.
        trend_valid=valid[...,1:,:]&valid[...,:-1,:]
        trend=(delta[...,1:,:]-delta[...,:-1,:]).abs()
        trend_err=(trend*trend_valid).sum((-1,-2))/trend_valid.sum((-1,-2)).clamp_min(1)
        coverage=denom/(h*a)
        margin=(coarse-coarse.mean(-1,keepdim=True))
        f=torch.stack([mae,trend_err,coverage,coarse,margin],-1)
        raw=coarse if mode=='no_mask_alignment' else (self.rerank(f).squeeze(-1)+coarse-.25*mae-.1*trend_err)
        ok=(denom>=1)&(coarse>-1e3)
        ranked,idloc=raw.masked_fill(~ok,-1e4).topk(min(self.topk,count),dim=-1)
        ids=torch.gather(id64,-1,idloc)
        selok=torch.gather(ok,-1,idloc)
        w=masked_softmax(ranked/self.log_temperature.exp().clamp_min(.02),selok)
        entropy=-(w*w.clamp_min(1e-9).log()).sum(-1)
        confidence=torch.stack([ctx['coverage'],ctx['missing'],selok.any(-1).float(),entropy],-1)
        return {'ids':ids,'weights':w,'valid':selok,'group_valid':selok.any(-1),'query':query,'confidence':confidence,'scores':ranked,'coverage':coverage.gather(-1,idloc)}
