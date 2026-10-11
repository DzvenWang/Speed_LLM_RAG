from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class RetrievedStateRecovery(nn.Module):
    """Top-K full historical target states + observable anchor-conditioned adaptation."""
    def __init__(self,hidden=64):
        super().__init__()
        self.adapter=nn.Sequential(nn.Linear(5,hidden),nn.GELU(),nn.Linear(hidden,2))
        self.gater=nn.Sequential(nn.Linear(6,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.obs_health=nn.Linear(hidden,1)
        nn.init.normal_(self.obs_health.weight,std=0.003)
        nn.init.zeros_(self.obs_health.bias)
        nn.init.zeros_(self.adapter[-1].weight);nn.init.zeros_(self.adapter[-1].bias)
        nn.init.constant_(self.gater[-1].bias,-1.5)
    def forward(self,batch,ctx,retrieval,regions,memory,observability=None):
        x=batch['x_obs'];m=batch['obs_mask'];b,h,n=x.shape;g=regions.g;k=retrieval['ids'].shape[-1]
        ids=retrieval.get('state_ids',retrieval['ids'])
        aid=regions.anchor_ids;ac=regions.a
        hx,hv=memory.fetch_nodes(ids,aid)  # [B,G,K,H,A]
        cm=ctx['anchor_mask'][:,:,None] & hv
        residual=(ctx['anchor_values'][:,:,None]-hx)
        anchor_delta=(residual*cm).sum(-1)/cm.sum(-1).clamp_min(1)
        anchor_var=((residual-anchor_delta[...,None]).square()*cm).sum(-1)/cm.sum(-1).clamp_min(1)
        anchor_cov=cm.float().mean(-1)
        # sqrt has an infinite derivative at zero.  Under severe incomplete
        # sensing there may be only one observable anchor, giving variance=0.
        # A positive floor is essential even if the forward value is finite.
        anchor_std=torch.sqrt(anchor_var.clamp_min(1e-6))
        feats=torch.stack([anchor_delta,anchor_std,anchor_cov,ctx['missing'][:,:,None,None].expand(b,g,k,h),retrieval['weights'][:,:,:,None].expand(b,g,k,h)],-1)
        shift,logscale=self.adapter(feats).unbind(-1) # [B,G,K,H]
        # Transfer observed candidate discrepancies to each target node via graph walks.
        aligned_residual=residual*cm.float()
        # Each node belongs to exactly one region: gather ONLY its region's
        # candidate evidence. Avoid a dense [B,G,K,H,N] intermediate.
        owner=regions.owner
        node_id=torch.arange(n,device=x.device)
        node_ids=ids[:,owner,:]  # [B,N,K]
        own_hist=memory.x[node_ids,: ,node_id[None,:,None]]  # [B,N,K,H]
        own_valid=memory.valid[node_ids,: ,node_id[None,:,None]]
        own_residual=aligned_residual[:,owner]  # [B,N,K,H,A]
        transfer=regions.graph_transfer[owner,node_id]  # [N,A]
        graph_delta=torch.einsum('bnkha,na->bnkh',own_residual,transfer)
        altered=own_hist+shift[:,owner]+graph_delta
        weights=retrieval['weights'][:,owner,:,None]*own_valid.float()
        denom=weights.sum(2).clamp_min(1e-6)
        weighted=(altered*weights).sum(2)/denom  # [B,N,H]
        var=((altered-weighted[:,:,None,:]).square()*weights).sum(2)/denom
        recovered=weighted.transpose(1,2)
        uncertainty=torch.sqrt(var.transpose(1,2).clamp_min(1e-6))
        coverage=ctx['coverage'][:,owner][:,None,:].expand(b,h,n)
        target_missing=ctx['missing'][:,owner][:,None,:].expand(b,h,n)
        exist=retrieval['group_valid'][:,owner][:,None,:].expand(b,h,n)
        # Missing retrieval support must not be reported as zero uncertainty.
        uncertainty=torch.where(exist,uncertainty,torch.full_like(uncertainty,10.0))
        if observability is None:
            obs_score=torch.zeros_like(x)
        else:
            obs_score=torch.tanh(self.obs_health(observability).squeeze(-1))[:,regions.owner][:,None,:].expand(b,h,n)
        inp=torch.stack([m.float(),coverage,target_missing,uncertainty.clamp(0,10),exist.float(),obs_score],-1)
        gate=torch.sigmoid(self.gater(inp)).squeeze(-1)*(~m).float()*exist.float()
        supported=torch.where(m,x,gate*recovered)
        return {'recovered':recovered,'uncertainty':uncertainty,'gate':gate,'supported':supported,'valid':exist}

class InnovationStudent(nn.Module):
    """Cross-attend target-node queries to Top-K historical latent tokens, decode rank factor."""
    def __init__(self,history,d=64,mem_dim=32,rank=16,heads=4):
        super().__init__();self.history=history;self.d=d
        self.memory_projection=nn.Linear(mem_dim,d)
        self.query_projection=nn.Linear(5,d)
        self.attention=nn.MultiheadAttention(d,heads,batch_first=True)
        self.lowrank=nn.Sequential(nn.Linear(d*2,2*d),nn.GELU(),nn.Linear(2*d,rank))
        self.time_basis=nn.Parameter(torch.randn(history,rank,d)*.01)
        self.local=nn.Sequential(nn.Linear(d*2+1,d),nn.GELU(),nn.Linear(d,d))
        self.uncertainty=nn.Sequential(nn.Linear(d,32),nn.GELU(),nn.Linear(32,1))
        nn.init.zeros_(self.lowrank[-1].weight);nn.init.zeros_(self.lowrank[-1].bias)
        nn.init.zeros_(self.local[-1].weight);nn.init.zeros_(self.local[-1].bias)
    def forward(self,batch,state,retrieval,regions,memory,base_hidden):
        b,h,n=batch['x_obs'].shape;k=retrieval['ids'].shape[-1];owner=regions.owner
        if memory.latent is not None:
            candidate=memory.latent[retrieval.get('latent_ids',retrieval['ids'])] # B,G,K,N,Dm
        else:
            # Missing latent cache is explicit; this baseline path retains history evidence.
            hx,_=memory.fetch(retrieval.get('latent_ids',retrieval['ids'])); candidate=hx.mean(-2)[...,None].expand(-1,-1,-1,-1,self.memory_projection.in_features)
        # Select the region appropriate to each target node without B,G,K,N duplication in attention.
        candidate=candidate.permute(0,3,1,2,4).gather(2,owner[None,:,None,None,None].expand(b,n,1,k,candidate.shape[-1])).squeeze(2)
        w=retrieval['weights'][:,owner,:] # B,N,K
        validity=retrieval['valid'][:,owner,:] # B,N,K
        node_state=state['recovered'].mean(1)
        qcov=batch['obs_mask'].float().mean(1)
        feat=torch.stack([node_state,qcov,state['uncertainty'].mean(1),w.max(-1).values,validity.any(-1).float()],-1)
        q=self.query_projection(feat)
        kv=self.memory_projection(candidate.float())*w[...,None]
        # all invalid -> do not feed fully masked keys to MHA; output later multiplied by valid.
        attended,_=self.attention(q.reshape(b*n,1,self.d),kv.reshape(b*n,k,self.d),kv.reshape(b*n,k,self.d),need_weights=False)
        attended=attended.reshape(b,n,self.d)*validity.any(-1)[...,None]
        bh=base_hidden.mean(1)
        features=torch.cat([bh,attended],-1)
        factors=self.lowrank(features) # B,N,R
        lowrank=torch.einsum('bnr,hrd->bhnd',factors,self.time_basis)
        res=self.local(torch.cat([base_hidden,attended[:,None].expand(b,h,n,self.d),batch['obs_mask'].float()[...,None]],-1))
        dz=(lowrank+res)*validity.any(-1)[:,None,:,None]
        u=F.softplus(self.uncertainty(attended)).squeeze(-1)[:,None,:].expand(b,h,n)
        return {'delta':dz,'uncertainty':u,'candidate_embedding':attended,'support_tokens':kv,'support_valid':validity}
