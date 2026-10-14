"""Permutation-aware set contextual candidate utility and adaptive Null selection."""
import torch
from torch import nn
import torch.nn.functional as F

class SetConditionalSelector(nn.Module):
    def __init__(self,d=64,horizon=12,final_k=8):
        super().__init__();self.final_k=final_k;self.p=horizon
        self.episode_embed=nn.Sequential(nn.Linear(9,d),nn.GELU(),nn.Linear(d,d))
        self.context_mix=nn.Sequential(nn.Linear(d*3,d),nn.GELU(),nn.Linear(d,d))
        self.benefit=nn.Sequential(nn.Linear(d+4,d),nn.GELU(),nn.Linear(d,2))
        self.null_head=nn.Sequential(nn.Linear(d+4,d),nn.GELU(),nn.Linear(d,1))
        self.horizon=nn.Embedding(horizon,d)
        self.uncertainty=nn.Linear(d,1)
    def forward(self,short,proposals,support,intervention=None):
        delta=proposals['deltas'];available=proposals['available'];b,n,k,types,p=delta.shape
        s=short['scores']
        r,e=delta[:,:,:,0,:],delta[:,:,:,1,:]
        avail_any=available.any(-1).any(-1)
        features=torch.stack([r.mean(-1),e.mean(-1),r.abs().mean(-1),e.abs().mean(-1),
                              r.std(-1,unbiased=False),e.std(-1,unbiased=False),
                              (r-e).abs().mean(-1),s.clamp(-1e4,1e4),avail_any.float()],-1)
        tokens=self.episode_embed(features)
        support_node=support['embedding'].mean(2)
        initial=s.masked_fill(~avail_any,-1e6).topk(min(self.final_k,k),dim=-1).indices
        selected_context=tokens.gather(2,initial[...,None].expand(-1,-1,-1,tokens.shape[-1])).mean(2)
        set_context=(tokens*avail_any[...,None]).sum(2)/avail_any.sum(2).clamp_min(1)[...,None]
        if intervention=='no_set_utility':
            selected_context=torch.zeros_like(selected_context);set_context=torch.zeros_like(set_context)
        context=self.context_mix(torch.cat([support_node,selected_context,set_context],-1))
        token=tokens+context[:,:,None,:]
        hz=self.horizon(torch.arange(p,device=delta.device))[None,None,None,:,:]
        h_token=token[:,:,:,None,:]+hz
        # [B,N,K,P,D] -> per-evidence-type benefit and uncertainty
        def heads(value):
            inp=torch.cat([h_token,value[...,None],value.abs()[...,None],
                           (r-e).abs()[...,None],s[:,:,:,None,None].expand(-1,-1,-1,p,-1)],-1)
            return self.benefit(inp)
        rscore=heads(r).permute(0,1,2,4,3)
        escore=heads(e).permute(0,1,2,4,3)
        scores=torch.stack([rscore[:,:,:,0,:],escore[:,:,:,0,:]],3)
        logits=torch.stack([rscore[:,:,:,1,:],escore[:,:,:,1,:]],3)
        uncertainty=F.softplus(self.uncertainty(h_token).squeeze(-1)).unsqueeze(3).expand_as(scores)
        # Calibrated utility mixes signed gain, benefit sign and predicted uncertainty.
        signed_scores=scores
        scores=signed_scores+0.15*logits.tanh()-0.05*uncertainty
        if intervention=='no_counterfactual_utility':
            scores=s[:,:,:,None,None].expand_as(scores)*0.1
        # Two-round set-conditioned episode selection; one controlled update.
        rank=scores.masked_fill(~available,-1e6).amax(3).mean(-1)
        count=min(k,self.final_k)
        chosen=rank.topk(count,-1).indices
        def gather(x):
            index=chosen[...,None,None].expand(-1,-1,-1,x.shape[-2],x.shape[-1])
            return x.gather(2,index)
        values=gather(delta);good=gather(available)
        chosen_scores=gather(scores);chosen_logits=gather(logits)
        # Query/node/horizon Null decision, based on predicted utility and uncertainty.
        disagree=(values.std(2,unbiased=False).mean(2)).permute(0,1,2) # [B,N,P]
        null_feat=torch.cat([support['embedding'],disagree[...,None],
                             chosen_scores.mean((2,3))[...,None],
                             chosen_logits.mean((2,3))[...,None],
                             (1-good.float().mean((2,3)))[...,None]],-1)
        null_logit=self.null_head(null_feat).squeeze(-1)
        if intervention=='no_null':null_logit=null_logit-10.0
        evidence_logits=chosen_scores.permute(0,1,4,2,3).reshape(b,n,p,count*types)
        evidence_valid=good.permute(0,1,4,2,3).reshape(b,n,p,count*types)
        evidence_logits=evidence_logits.masked_fill(~evidence_valid,-1e7)
        w=torch.cat([evidence_logits,null_logit[...,None]],-1).softmax(-1)
        weights=w[...,:count*types].view(b,n,p,count,types).permute(0,1,3,4,2)
        null=w[...,-1]
        correction=(values*weights).sum((2,3))
        # pooled expected signed gain; safe when all candidates are invalid
        expected=(chosen_scores*weights).sum((2,3))
        return {'selected_ids':short['ids'].gather(2,chosen),'selected_positions':chosen,
                'weights':weights,'null':null,'correction':correction,
                'predicted_gain':expected,'all_utility':scores,'all_benefit_logit':logits,
                'all_uncertainty':uncertainty,'selected_values':values,'selected_valid':good}
