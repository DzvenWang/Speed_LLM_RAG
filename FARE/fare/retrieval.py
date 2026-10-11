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
        # Dynamic failure observability belongs to the QUERY, not the static memory key.
        self.observability_query=nn.Linear(hidden,key_dim,bias=False)
        nn.init.normal_(self.observability_query.weight,std=0.003)
    def _features(self,x,valid,extra):
        # [B,G,H,A] -> [B,G,2H+2], with validity masking.
        m=valid.float(); v=(x*m).sum(-1)/m.sum(-1).clamp_min(1)
        c=m.mean(-1)
        return torch.cat([v,c,extra],-1)
    def query(self,ctx,observability=None):
        f=self._features(ctx['anchor_values'],ctx['anchor_mask'],torch.stack([ctx['missing'],ctx['coverage']],-1))
        key=self.q(f)
        if observability is not None:
            key=key+self.observability_query(observability)
        return F.normalize(key,dim=-1)
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
        self.utility_head=nn.Sequential(nn.Linear(7,hidden),nn.GELU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.utility_head[-1].weight)
        nn.init.zeros_(self.utility_head[-1].bias)
        self.log_temperature=nn.Parameter(torch.tensor(math.log(.3)))
    @torch.no_grad()
    def refresh(self,memory,regions):
        was_training=self.encoder.training;self.encoder.eval()
        memory.keys=self.encoder.memory_keys(memory,regions)
        if memory.future is not None:
            memory.ensure_region_future(regions)
        self.encoder.train(was_training)
    def forward(self,ctx,regions,memory,window_start,mode='full',observability=None,diagnostics=False,demand=None,base_forecast=None):
        if memory.keys is None:raise RuntimeError('Memory index not built; call retriever.refresh')
        b,g,h,a=ctx['anchor_values'].shape
        query=self.encoder.query(ctx,observability)
        if demand is not None:
            query=F.normalize(query+0.1*torch.tanh(demand),dim=-1)
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
        # Direct sparse gather: [B,G,K,H,A], never [B,G,K,H,N].
        ar,av=memory.fetch_nodes(id64,regions.anchor_ids)
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
        if mode=='no_mask_alignment':
            raw=coarse
        elif mode=='no_learned_reranker':
            raw=coarse-.25*mae-.1*trend_err
        elif mode=='no_observation_distance':
            raw=self.rerank(f).squeeze(-1)+coarse
        else:
            raw=self.rerank(f).squeeze(-1)+coarse-.25*mae-.1*trend_err
        # Historical future trajectories are allowed only when their entire
        # forecast horizon had finished before the current query was issued.
        # Forecast target labels NEVER enter the inference features.
        analog=None
        if memory.future is not None:
            horizon=memory.future.shape[1]
            current_issue=window_start+self.overlap-horizon
            legal_future=memory.starts[id64]+self.overlap<=current_issue[:,None,None]
            raw=raw.masked_fill(~legal_future,-1e4)
            # Valid-only region aggregates have already been cached by refresh().
            # Exact same regional-mean evidence, without [B,G,K,P,N] gathers.
            if memory.region_future is None:
                memory.ensure_region_future(regions)
            rg=torch.arange(g,device=id64.device)[None,:,None]
            analog=memory.region_future[id64,rg]
            if base_forecast is not None:
                # Mean and trend agreement are observable-model forecasts, not labels.
                breg=torch.einsum('bpn,gn->bgp',base_forecast,regions.target_mask)/regions.target_mask.sum(-1)[None,:,None].clamp_min(1)
                d=(analog-breg[:,:,None,:]).abs().mean(-1)
                features=torch.stack([mae,trend_err,coverage,coarse,margin,d,analog.std(-1,unbiased=False)],-1)
                raw=raw+self.utility_head(features).squeeze(-1)
        ok=(denom>=1)&(coarse>-1e3)&(raw>-1e3)
        ranked,idloc=raw.masked_fill(~ok,-1e4).topk(min(self.topk,count),dim=-1)
        ids=torch.gather(id64,-1,idloc)
        selok=torch.gather(ok,-1,idloc)
        w=masked_softmax(ranked/self.log_temperature.exp().clamp_min(.02),selok)
        entropy=-(w*w.clamp_min(1e-9).log()).sum(-1)
        confidence=torch.stack([ctx['coverage'],ctx['missing'],selok.any(-1).float(),entropy],-1)
        out={'ids':ids,'weights':w,'valid':selok,'group_valid':selok.any(-1),
             'query':query,'confidence':confidence,'scores':ranked,'coverage':coverage.gather(-1,idloc),'utility_scores':ranked}
        if analog is not None:
            out['analog_future']=analog.gather(-2,idloc[...,None].expand(b,g,ids.shape[-1],analog.shape[-1]))
        if diagnostics:
            out.update({'coarse_ids':id64,'coarse_scores':coarse,'match_error':mae.gather(-1,idloc),
                        'trend_error':trend_err.gather(-1,idloc)})
        return out
