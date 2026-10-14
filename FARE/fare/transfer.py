"""Graph-neighborhood, regime-conditioned lag-aware dual evidence transport."""
import torch
from torch import nn
import torch.nn.functional as F

class PropagationTransport(nn.Module):
    def __init__(self,d=64,horizon=12):
        super().__init__();self.p=horizon
        self.kernel=nn.Sequential(nn.Linear(d+5,d),nn.GELU(),nn.Linear(d,3))
        self.residual_map=nn.Sequential(nn.Linear(d+4,d),nn.GELU(),nn.Linear(d,1))
        self.evolution_map=nn.Sequential(nn.Linear(d+4,d),nn.GELU(),nn.Linear(d,1))
    def forward(self,short,memory,graph,support,base_prediction,intervention=None):
        ids=short['ids'];valid=short['valid'];device=ids.device
        b,n,k=ids.shape;p=base_prediction.shape[1]
        residual=memory.fetch_node(ids,'residual',device)
        resid_valid=memory.fetch_node(ids,'residual_valid',device).bool()
        future=memory.fetch_node(ids,'future',device)
        future_valid=memory.fetch_node(ids,'future_valid',device).bool()
        history=memory.fetch_node(ids,'x',device)
        history_valid=memory.fetch_node(ids,'mask',device).bool()
        # robust historical last valid observed value, not an invalid trailing padding value
        timeline=torch.arange(history.shape[-1],device=device)
        times=torch.where(history_valid,timeline[None,None,None,:],-1).amax(-1)
        historical_level=history.gather(-1,times.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        historical_level=torch.where(times>=0,historical_level,torch.zeros_like(historical_level))
        # Graph evidence: gather only L neighbor values and aggregate with fixed legal topology
        neigh=memory.fetch_neighbors(ids,graph.neighbor_ids,'future',device) # [B,N,K,L,P]
        nw=graph.neighbor_weight[None,:,None,:,None]
        neigh_mean=(neigh*nw).sum(3)
        support_embed=support['embedding'] # [B,N,P,D]
        regime=support['regime']
        boundary=torch.stack([support['forward_coverage'],support['backward_coverage'],
                              support['missingness'][:,:,None].expand(-1,-1,p),
                              base_prediction.permute(0,2,1),
                              (neigh_mean-future).mean(2)],-1)
        kernel_logits=self.kernel(torch.cat([support_embed,boundary],-1))
        if intervention in ('no_alignment','no_direction'):
            lag_weight=torch.tensor([1.,0.,0.],device=device,dtype=kernel_logits.dtype).view(1,1,1,3).expand(b,n,p,-1)
        else:lag_weight=kernel_logits.softmax(-1)
        # Strictly no wraparound at horizon boundary. Candidate lag is 0,1,2.
        def align(v):
            lagged=torch.stack([v,F.pad(v[...,:-1],(1,0)),F.pad(v[...,:-2],(2,0))],-1)
            return (lagged*lag_weight[:,:,None,:,:]).sum(-1)
        aligned_residual=align(residual)
        aligned_future=align(.75*future+.25*neigh_mean)
        # Evolution is a relative change from candidate's last observable historical point.
        evolution=aligned_future-historical_level[...,None]
        forecast_delta=base_prediction.permute(0,2,1)-base_prediction[:,0,:][:,:,None]
        ctx=support_embed[:,:,None,:,:].expand(-1,-1,k,-1,-1)
        z_res=torch.cat([ctx,aligned_residual[...,None],evolution[...,None],
                         forecast_delta[:,:,None,:,None].expand(-1,-1,k,-1,-1),
                         (aligned_residual-evolution)[...,None]],-1)
        z_evo=torch.cat([ctx,evolution[...,None],aligned_residual[...,None],
                         forecast_delta[:,:,None,:,None].expand(-1,-1,k,-1,-1),
                         (evolution-forecast_delta[:,:,None,:])[...,None]],-1)
        # bounded direct residual identity + trainable signed modulation
        r_delta=aligned_residual*(1+.25*self.residual_map(z_res).squeeze(-1).tanh())
        e_delta=(evolution-forecast_delta[:,:,None,:])*(1+.25*self.evolution_map(z_evo).squeeze(-1).tanh())
        v_r=valid[...,None]&resid_valid
        v_e=valid[...,None]&future_valid&(times[...,None]>=0)
        if intervention=='no_residual':v_r=torch.zeros_like(v_r)
        if intervention=='no_evolution':v_e=torch.zeros_like(v_e)
        deltas=torch.stack([r_delta,e_delta],3)  # [B,N,K,2,P]
        availability=torch.stack([v_r,v_e],3)
        agreement=1-(r_delta-e_delta).abs()/(1+r_delta.abs()+e_delta.abs())
        return {'deltas':deltas,'available':availability,'alignment_confidence':lag_weight[...,0],
                'agreement':agreement,'raw_residual':aligned_residual,'evolution':evolution}
