"""Permutation-equivariant, interaction-aware Keep/Swap utility and adaptive Null.

Set interaction occurs over K candidate tokens, never K×H×d full attention.
Hard discrete swaps receive explicit train-only counterfactual supervision.
"""
import torch
from torch import nn
import torch.nn.functional as F

class SetConditionalSelector(nn.Module):
    def __init__(self,d=64,horizon=12,final_k=8,heads=4):
        super().__init__();self.final_k=int(final_k);self.horizon=horizon
        self.embed=nn.Sequential(nn.Linear(12,d),nn.GELU(),nn.Linear(d,d))
        self.demand=nn.Linear(d,d)
        self.attn1=nn.MultiheadAttention(d,heads,batch_first=True)
        self.attn2=nn.MultiheadAttention(d,heads,batch_first=True)
        self.norm1=nn.LayerNorm(d);self.norm2=nn.LayerNorm(d)
        self.keep=nn.Linear(d,2)
        self.benefit=nn.Linear(d,2)
        self.uncertainty=nn.Sequential(nn.Linear(d,2),nn.Softplus())
        self.time=nn.Embedding(horizon,d)
        self.time_proj=nn.Linear(d,2)
        self.swap_q=nn.Linear(d,d,bias=False)
        self.swap_k=nn.Linear(d,d,bias=False)
        self.swap_bias=nn.Sequential(nn.Linear(4,d//2),nn.GELU(),nn.Linear(d//2,1))
        self.null_head=nn.Sequential(nn.Linear(d+4,d),nn.GELU(),nn.Linear(d,1))
        self.swap_threshold=nn.Parameter(torch.tensor(0.05))

    def forward(self,short,proposals,support,intervention=None,forced_initial=None):
        delta=proposals['deltas'];valid=proposals['available'];b,n,k,t,p=delta.shape
        r,e=delta[:,:,:,0,:],delta[:,:,:,1,:]
        availability=valid.any(-1).any(-1)&short['valid']
        state_score=short['scores'].clamp(-10,10)
        cmp=proposals['compatibility']
        dmd=support['demand']
        feat=torch.stack([r.mean(-1),e.mean(-1),r.abs().mean(-1),e.abs().mean(-1),
                          r.std(-1,unbiased=False),e.std(-1,unbiased=False),
                          (r-e).abs().mean(-1),state_score,availability.float(),
                          cmp.mean(-1),valid[:,:,:,0].float().mean(-1),
                          valid[:,:,:,1].float().mean(-1)],-1)
        tok=self.embed(feat)+self.demand(support['embedding'].mean(2))[:,:,None,:]
        tok=tok*availability[...,None]
        if intervention not in ('no_set_utility','no_set_interaction'):
            view=tok.reshape(b*n,k,-1)
            safe=availability.reshape(b*n,k).clone()
            safe[~safe.any(-1),0]=True
            att,_=self.attn1(view,view,view,key_padding_mask=~safe,need_weights=False)
            view=self.norm1(view+att)
            att,_=self.attn2(view,view,view,key_padding_mask=~safe,need_weights=False)
            tok=self.norm2(view+att).reshape(b,n,k,-1)
            tok=tok*availability[...,None]
        # Per-node candidate tokens -> lightweight Horizon-conditioned type heads.
        logits0=self.keep(tok)                                    # [B,N,K,2]
        benefit0=self.benefit(tok)
        unc0=self.uncertainty(tok)
        time=self.time_proj(self.time(torch.arange(p,device=tok.device)))
        signed=logits0[...,None]+time[None,None,None,:,:].permute(0,1,2,4,3)
        benefit=benefit0[...,None].expand_as(signed)
        uncertainty=unc0[...,None].expand_as(signed)
        signed=signed+0.25*delta-0.1*delta.abs()+0.1*cmp[:,:,:,None,:]
        signed=signed+0.1*benefit.tanh()-0.05*uncertainty
        if intervention in ('no_counterfactual_utility','state_similarity_only'):
            signed=short['scores'][:,:,:,None,None].expand_as(signed)*0.1
        # Do not rank using invalid horizon/type evidence.
        hard=signed.masked_fill(~valid,-1e6)
        rank=hard.amax(3).masked_fill(~valid.any(3),0).mean(-1)
        rank=rank.masked_fill(~availability,-1e7)
        take=min(k,self.final_k)
        initial=(forced_initial.to(rank.device).long() if forced_initial is not None
                 else rank.topk(take,-1).indices)
        # Explicit one-step legal KEEP/SWAP policy.
        # Pairwise low-rank interaction [B,N,selected,all], not [B,N,selected,all,d*2].
        sel_token=tok.gather(2,initial[...,None].expand(-1,-1,-1,tok.shape[-1]))
        q=self.swap_q(sel_token);key=self.swap_k(tok)
        pair=torch.einsum('bnjd,bnkd->bnjk',q,key)/(tok.shape[-1]**.5)
        # Additional overlap/conflict from mean Residual/Evolution signatures.
        sel_r=r.mean(-1).gather(2,initial)
        sel_e=e.mean(-1).gather(2,initial)
        all_r=r.mean(-1);all_e=e.mean(-1)
        pair_features=torch.stack([sel_r[...,None]-all_r[:,:,None,:],
                                   sel_e[...,None]-all_e[:,:,None,:],
                                   (sel_r[...,None]-all_r[:,:,None,:]).abs(),
                                   (sel_e[...,None]-all_e[:,:,None,:]).abs()],-1)
        pair=pair+self.swap_bias(pair_features).squeeze(-1)
        already=(initial[...,None]==torch.arange(k,device=tok.device)[None,None,None,:]).any(2)
        legal=availability&(~already)
        pair=pair.masked_fill(~legal[:,:,None,:],-1e7)
        selected_valid=availability.gather(2,initial)
        pair=pair.masked_fill(~selected_valid[...,None],-1e7)
        chosen=initial.clone()
        if intervention not in ('no_swap_utility','no_set_utility','no_set_interaction','no_counterfactual_utility'):
            flat=pair.reshape(b,n,-1)
            best,best_idx=flat.max(-1)
            selected_slot=torch.div(best_idx,k,rounding_mode='floor')
            candidate_pos=best_idx%k
            perform=best>self.swap_threshold.abs()
            chosen.scatter_(2,selected_slot[...,None],torch.where(perform[...,None],candidate_pos[...,None],
                            initial.gather(2,selected_slot[...,None])))
        def gather(z):
            return z.gather(2,chosen[...,None,None].expand(-1,-1,-1,z.shape[-2],z.shape[-1]))
        vals=gather(delta);good=gather(valid)
        sc=gather(signed);benefit_sel=gather(benefit)
        disagreement=vals.std(2,unbiased=False).mean(2)
        null_features=torch.cat([support['embedding'],disagreement[...,None],
                                 sc.mean((2,3))[...,None],
                                 dmd[...,None],(1-good.float().mean((2,3)))[...,None]],-1)
        null_logits=self.null_head(null_features).squeeze(-1)
        if intervention=='no_null':null_logits=null_logits-10
        flat=sc.permute(0,1,4,2,3).reshape(b,n,p,-1)
        active=good.permute(0,1,4,2,3).reshape(b,n,p,-1)
        flat=flat.masked_fill(~active,-1e7)
        weights=torch.cat([flat,null_logits[...,None]],-1).softmax(-1)
        w=weights[...,:take*t].view(b,n,p,take,t).permute(0,1,3,4,2)
        null=weights[...,-1]
        expected=(sc*w).sum((2,3))
        return {'selected_ids':short['ids'].gather(2,chosen),'selected_positions':chosen,
                'weights':w,'null':null,'correction':(vals*w).sum((2,3)),
                'predicted_gain':expected,'all_utility':signed,'all_benefit_logit':benefit,
                'all_uncertainty':uncertainty,'selected_values':vals,'selected_valid':good,
                'initial_positions':initial,'swap_scores':pair,'candidate_tokens':tok,
                'keep_scores':signed}
