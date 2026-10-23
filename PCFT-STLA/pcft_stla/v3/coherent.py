"""PCFT v3 historical propagation with *one complete Train episode per path*.

Never splice different historical time windows into an edge. Region-selected
episodes are fetched in full before propagation, not by per-node concatenation.
"""
from __future__ import annotations
import torch
from torch import nn


class EpisodeCoherentPropagation(nn.Module):
    def __init__(self, horizon: int, hidden=32):
        super().__init__()
        self.horizon=int(horizon)
        self.scale=nn.Sequential(nn.Linear(5,hidden),nn.GELU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.scale[-1].weight)
        nn.init.zeros_(self.scale[-1].bias)

    def forward(self,propagation, global_x,global_y,regional_x,regional_y,
                query_x,query_mask, candidate_weights, regional_assignment,
                region_index, support, mode='full'):
        """global [B,G,H,N], regional [B,R,K,H,N], region_index [B,R,K].

        Candidate weights are [B,G+K,P,N], matching the v2 scorer's *node-local*
        region ranks. Propagation is constructed from unspliced full episodes.
        """
        b,g,h,n=global_x.shape
        p=self.horizon
        _,r,k,_,_=regional_x.shape
        if region_index.shape!=(b,r,k):raise ValueError('Regional provenance IDs must be [B,R,K]')
        if candidate_weights.shape!=(b,g+k,p,n):raise ValueError('Candidate weight dimensional mismatch')
        # Flatten the actual full Train episodes, not node-wise stitched cases.
        original=torch.cat([global_x,regional_x.reshape(b,r*k,h,n)],dim=1)
        fut=torch.cat([global_y,regional_y.reshape(b,r*k,p,n)],dim=1)
        if mode=='no_lag_alignment':
            # Disable alignment ONLY, preserve the graph and all action code.
            lag_override=0
        else:lag_override=None
        # Implement the graph/lag ablations in the actual propagation operator.
        all_prop=propagation.transfer(original,fut,p,query_x=query_x,query_mask=query_mask,
                                      provenance=None,uniform_graph=(mode=='no_directional_graph'),
                                      lag_override=lag_override)
        gp=all_prop[:,:g]
        rp=all_prop[:,g:].reshape(b,r,k,p,n)
        region_map=regional_assignment.long()
        if region_map.shape!=(n,):raise ValueError('Region node assignment must be [N]')
        node_region=region_map[None,None,None,None,:].expand(b,1,k,p,n)
        # Pick *one full episode* for each target node's region and rank.
        selected_region=torch.gather(rp,1,node_region).squeeze(1)
        pp=torch.cat([gp,selected_region],dim=1)
        raw=(pp.float()*candidate_weights.float()).sum(1)
        # Direct observation support and inferred future support remain distinct.
        observed=support['observed'].float()
        forecast=support['forecast'].float()
        uncertainty=support['uncertainty'].float()
        feature=torch.stack([raw.abs(),observed,forecast,uncertainty,
                             query_mask.float().mean(1)[:,None].expand(b,p,n)],-1)
        reliability=torch.sigmoid(self.scale(feature).squeeze(-1))
        # Start close to raw propagation, but confidence-attenuate when uncertain.
        trusted=(reliability * (observed+forecast*(1-uncertainty)).clamp(0,1))
        return {'propagation_correction':raw*trusted,
                'raw_propagation':raw,'propagation_availability':trusted,
                'episode_count':g+r*k}
