"""Three-stage hierarchical, forecast-conditioned retrieval.

Region Top64 -> node-specific Top16 -> forecast-utility Top8 + null.
The hard candidate-id selection is supervised with a separate differentiable
query/key contrastive objective; selected weights receive forecast gradients.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .support_query import ForecastQueryEncoder
from .evidence_adapter import MaskConditionedEvidenceAdapter,CandidateForecastProbe


def safe_softmax(scores,valid,dim=-1):
    safe=scores.masked_fill(~valid,-1e4)
    values=torch.softmax(safe,dim=dim)*valid.float()
    return values/values.sum(dim=dim,keepdim=True).clamp_min(1e-8)


class HierarchicalForecastRetriever(nn.Module):
    def __init__(self,history,horizon,base_dim,hidden=64,key_dim=128,
                 coarse_topk=64,shortlist_topk=16,topk=8,latent_dim=32,overlap=24):
        super().__init__()
        self.encoder=ForecastQueryEncoder(history,base_dim,hidden,key_dim)
        self.coarse_topk=coarse_topk;self.shortlist_topk=shortlist_topk
        self.topk=topk;self.overlap=overlap
        self.fast=nn.Sequential(nn.Linear(key_dim+latent_dim+4,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.utility=nn.Sequential(nn.Linear(hidden+key_dim+4,hidden),nn.GELU(),nn.Linear(hidden,2))
        self.adapter=MaskConditionedEvidenceAdapter(history,horizon,latent_dim,hidden)
        self.probe=CandidateForecastProbe(hidden,horizon)
        self.memory_latent_projection=nn.Linear(latent_dim,key_dim,bias=False)
        self.log_temperature=nn.Parameter(torch.tensor(math.log(.5)))
        self.null_logit=nn.Parameter(torch.tensor(-2.0))
        self.latent_dim=latent_dim

    @staticmethod
    def _support_signature(values,valid,regions):
        """Masked upstream/downstream/all-anchor trends, preserving order.

        Shape: [B,G,H,A] -> [B,G,4H], used only for diversified recall.
        """
        m=valid.float()
        up=regions.up_anchor[None,:,None,:]
        down=regions.down_anchor[None,:,None,:]
        def reduce(weights):
            weight=m*weights
            return (values*weight).sum(-1)/weight.sum(-1).clamp_min(1)
        allmean=reduce(torch.ones_like(up))
        upstream=reduce(up)
        downstream=reduce(down)
        trend=torch.cat([allmean[:,:,1:]-allmean[:,:,:-1],
                         torch.zeros_like(allmean[:,:,:1])],-1)
        return F.normalize(torch.cat([upstream,downstream,allmean,trend],-1),dim=-1)

    @torch.no_grad()
    def _memory_signature(self,memory,regions,batch=256):
        ids=regions.anchor_ids.flatten()
        result=[]
        for lo in range(0,len(memory.x),batch):
            x=memory.x[lo:lo+batch];v=memory.valid[lo:lo+batch]
            b,h,n=x.shape
            xx=x[:,:,ids].reshape(b,h,regions.g,regions.a).permute(0,2,1,3)
            vv=v[:,:,ids].reshape(b,h,regions.g,regions.a).permute(0,2,1,3)&regions.anchor_valid[None,:,None,:]
            result.append(self._support_signature(xx,vv,regions))
        return torch.cat(result,0)

    def _mixed_candidates(self,learned,observed,trend,legal,window_start,count,mode):
        """Top candidates from learned/masked/trend/exploration, de-duplicated.

        Falls back to the learned pool when a source has no useful observed
        support. All selected IDs are still checked against the causal mask.
        """
        b,g,m=learned.shape
        base_ids=learned.topk(count,-1).indices
        if mode=='random_memory':
            grid=torch.arange(m,device=learned.device,dtype=torch.float32)[None,None,:]
            seed=window_start.float()[:,None,None]
            rnd=torch.frac(torch.sin((grid+1)*12.9898+(seed+1)*78.233)*43758.5453)
            return rnd.expand(b,g,-1).masked_fill(~legal[:,None,:],-1e4).topk(count,-1).indices
        quotas=[max(1,round(count*.55)),max(1,round(count*.25)),max(1,round(count*.12))]
        quotas=[min(count,int(q)) for q in quotas]
        # The final exploration quota is bounded by the remaining candidate
        # budget; learned fallback guarantees 64 distinct entries if possible.
        exploration=max(0,count-sum(quotas))
        ranked=[learned.topk(quotas[0],-1).indices,
                observed.topk(quotas[1],-1).indices,
                trend.topk(quotas[2],-1).indices]
        if exploration:
            grid=torch.arange(m,device=learned.device,dtype=torch.float32)[None,None,:]
            seed=window_start.float()[:,None,None]
            noise=torch.frac(torch.sin((grid+1)*35.123+(seed+1)*71.933)*45455.545)
            ranked.append(noise.expand(b,g,-1).masked_fill(~legal[:,None,:],-1e4).topk(exploration,-1).indices)
        ids=torch.cat([*ranked,base_ids],-1)
        width=ids.shape[-1]
        # First occurrence wins. Select earliest 'count' unique valid IDs.
        earlier=torch.arange(width,device=ids.device)
        duplicates=(ids[..., :,None]==ids[...,None,:]) & \
                   (earlier[None,None,None,:]<earlier[None,None,:,None])
        fresh=~duplicates.any(-1)
        selected_position=torch.arange(width,device=ids.device)[None,None,:].expand(b,g,-1)
        select_score=torch.where(fresh,-selected_position.float(),torch.full_like(selected_position.float(),-1e6))
        positions=select_score.topk(count,-1).indices
        return ids.gather(-1,positions)

    @torch.no_grad()
    def refresh(self,memory,regions):
        was=self.encoder.training;self.encoder.eval()
        memory.keys=self.encoder.memory_keys(memory,regions)
        memory.source_signature=self._memory_signature(memory,regions)
        self.encoder.train(was)

    def _legal(self,memory,window_start):
        # Global window positions. Outcome of historical episode must be known
        # strictly before the current query's issue time.
        query_issue=window_start[:,None]+self.overlap-memory.future.shape[1]
        end=memory.starts[None,:]+self.overlap
        return ((window_start[:,None]-memory.starts[None,:]).abs()>=self.overlap) & (end<=query_issue)

    def forward(self,inputs,base,ctx,regions,memory,mode='full'):
        if memory.keys is None:raise RuntimeError('Refresh historical memory keys before retrieval')
        b,h,n=inputs['x_obs'].shape
        qry=self.encoder(ctx,regions,base['hidden'],base['prediction'],inputs['x_obs'],inputs['obs_mask'])
        g=regions.g;M=len(memory.x)
        sim=torch.einsum('bgd,mgd->bgm',qry['region_query'],memory.keys.detach())
        legal=self._legal(memory,inputs['window_start'])
        sim=sim.masked_fill(~legal[:,None,:],-1e4)
        count=min(self.coarse_topk,M)
        signature=self._support_signature(ctx['anchor_values'],ctx['anchor_mask'],regions)
        observed=torch.einsum('bgd,mgd->bgm',signature,memory.source_signature.detach())
        observed=observed.masked_fill(~legal[:,None,:],-1e4)
        # Trend-only retrieval uses the final H dimensions of the signature.
        tsize=ctx['anchor_values'].shape[2]
        trend=torch.einsum('bgd,mgd->bgm',signature[...,-tsize:],
                                  memory.source_signature[...,-tsize:].detach())
        trend=trend.masked_fill(~legal[:,None,:],-1e4)
        ids=self._mixed_candidates(sim,observed,trend,legal,inputs['window_start'],count,mode)
        scores=sim.gather(-1,ids)
        owner=regions.owner
        node_ids=ids[:,owner,:]
        coarse_score=scores[:,owner,:]
        node_index=torch.arange(n,device=ids.device)[None,:,None]
        node_index=node_index.expand(b,n,count)
        candidate_latent=memory.fetch_target_latent(node_ids,self.latent_dim)
        candidate_last=memory.x[node_ids,-1,node_index]
        cand_last_valid=memory.valid[node_ids,-1,node_index]
        current_last=inputs['x_obs'][:,-1,:,None]
        cur_valid=inputs['obs_mask'][:,-1,:,None]
        observed_delta=(candidate_last-current_last).abs()*cur_valid.float()*cand_last_valid.float()
        cov=memory.valid[node_ids,:,node_index].float().mean(-1)
        qn=qry['node_query'][:,:,None,:].expand(-1,-1,count,-1)
        latent_proj=F.normalize(self.memory_latent_projection(candidate_latent),dim=-1)
        align=(qn*latent_proj).sum(-1)
        valid=legal.gather(1,node_ids.reshape(b,-1)).reshape(b,n,count)&(coarse_score>-1e3)
        f=torch.cat([qn,candidate_latent,coarse_score[...,None],align[...,None],
                     observed_delta[...,None],cov[...,None]],-1)
        fast_scores=self.fast(f).squeeze(-1)+.2*coarse_score+.2*align-.1*observed_delta
        if mode=='similarity_memory':fast_scores=coarse_score
        short_k=min(self.shortlist_topk,count)
        short_scores,short_index=fast_scores.masked_fill(~valid,-1e4).topk(short_k,-1)
        short_ids=node_ids.gather(-1,short_index)
        short_valid=valid.gather(-1,short_index)
        short_query=qry['node_query'][:,:,None,:].expand(-1,-1,short_k,-1)
        evidence=self.adapter(memory,short_ids,inputs,base['prediction'],short_query,self.latent_dim)
        # Top16 shortlist receives a separate, candidate-specific forecast probe.
        probe_predictions=self.probe(base['prediction'],evidence)
        rel=evidence['reliability']
        effective=short_valid&evidence['future_valid'].any(-1)
        utility_feat=torch.cat([evidence['token'],short_query,
                 short_scores[...,None],rel[...,None],
                 qry['node_coverage'][:,:,None,None].expand(-1,-1,short_k,-1),
                 evidence['forecast_delta'].abs().mean(-1,keepdim=True)],-1)
        utility=self.utility(utility_feat)
        utility_score=utility[...,0]
        utility_risk=F.softplus(utility[...,1])
        if mode in ('utility_off','similarity_memory'):
            ranking=short_scores
        else:
            ranking=short_scores+.75*utility_score-.15*utility_risk
        k=min(self.topk,short_k)
        best,which=ranking.masked_fill(~effective,-1e4).topk(k,-1)
        selected_valid=effective.gather(-1,which)
        selected_ids=short_ids.gather(-1,which)
        # Append NULL to softmax normalization: explicit abstention.
        logits=torch.cat([best/self.log_temperature.exp().clamp_min(.05),
                self.null_logit.expand(b,n,1)],-1)
        logit_valid=torch.cat([selected_valid,torch.ones((b,n,1),device=ids.device,dtype=torch.bool)],-1)
        weights_all=safe_softmax(logits,logit_valid)
        if mode=='weight_shuffle':
            weights_all=torch.cat([weights_all[...,:-1].roll(1,-1),weights_all[...,-1:]],-1)
        weights=weights_all[...,:-1]
        null_weight=weights_all[...,-1]
        # Keep top16 evidence for genuine candidate-level utility supervision.
        # Gather final K evidence without rescanning the historical memory.
        chosen={}
        for name,tensor in evidence.items():
            chosen[name]=torch.gather(tensor,2,which[(...,)+(None,)*(tensor.ndim-3)].expand(*which.shape,*tensor.shape[3:]))
        selected_delta=(chosen['forecast_delta']*weights[...,None]).sum(2)
        tokens=chosen['token']*weights[...,None]
        if mode in ('null_evidence','retrieval_off'):
            selected_delta=torch.zeros_like(selected_delta)
            tokens=torch.zeros_like(tokens)
            weights=torch.zeros_like(weights)
            null_weight=torch.ones_like(null_weight)
        if mode=='cross_query_shuffle' and b>1:
            # Perform legal cross-query replacement (recomputed eligibility).
            # Shuffle from another batch query, same graph target and target node.
            shifted=selected_ids.roll(1,0)
            issue=inputs['window_start'][:,None,None]+h
            causal=(memory.starts[shifted]+h+memory.future.shape[1]<=issue)
            overlap=(inputs['window_start'][:,None,None]-memory.starts[shifted]).abs()>=self.overlap
            can_use=causal & overlap & selected_valid.roll(1,0) & selected_valid
            replacement=self.adapter(memory,shifted,inputs,base['prediction'],short_query[:,:,:k],self.latent_dim)
            # Every downstream evidence field, not just the forecast delta,
            # must follow the swapped historical ID to avoid false attribution.
            for name,old in chosen.items():
                newer=replacement[name]
                predicate=can_use[(...,)+(None,)*(old.ndim-3)]
                chosen[name]=torch.where(predicate,newer,old)
            tokens=chosen['token']*weights[...,None]
            selected_delta=(chosen['forecast_delta']*weights[...,None]).sum(2)
            selected_ids=torch.where(can_use,shifted,selected_ids)
        return {
            'query':qry,'ids':selected_ids,'weights':weights,'valid':selected_valid,
            'node_valid':selected_valid.any(-1),'null_weight':null_weight,
            'tokens':tokens,'forecast_delta':selected_delta,
            'utility':utility_score.gather(-1,which),
            'risk':utility_risk.gather(-1,which),
            'coarse_ids':ids,'coarse_scores':scores,
            'coarse_observed_scores':observed.gather(-1,ids),
            'shortlist_ids':short_ids,'shortlist_valid':effective,
            'shortlist_scores':ranking,'shortlist_weights':safe_softmax(ranking,effective),
            'probe_predictions':probe_predictions,
            'shortlist_utility':utility_score,
            'shortlist_risk':utility_risk,
            'selected_evidence':chosen,
            'node_query':qry['node_query'],
            'region_query':qry['region_query']
        }
