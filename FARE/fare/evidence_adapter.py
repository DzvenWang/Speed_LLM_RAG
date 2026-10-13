"""Sparse target-node historical evidence with mask-conditioned future adaptation."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class MaskConditionedEvidenceAdapter(nn.Module):
    def __init__(self, history=12, horizon=12, memory_dim=32, hidden=64):
        super().__init__()
        self.history=history;self.horizon=horizon
        self.history_encoder=nn.Sequential(nn.Linear(2*history+2,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.future_encoder=nn.Sequential(nn.Linear(horizon*2+hidden+memory_dim+2,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.delta_head=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,horizon))
        self.reliability_head=nn.Sequential(nn.Linear(hidden+2,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.state_head=nn.Linear(hidden,history)
        # A non-parametric future tendency is retained; learned component starts small.
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(self,memory,ids,inputs,base_forecast,node_query,latent_dim,node_ids=None):
        # ids [B,N,K]. Sparse gathers materialize only the selected target node.
        hist,hvalid=memory.fetch_target(ids,'history',node_ids=node_ids)
        future,fvalid=memory.fetch_target(ids,'future',node_ids=node_ids)
        latent=memory.fetch_target_latent(ids,latent_dim,node_ids=node_ids)
        b,n,k,h=hist.shape;p=future.shape[-1]
        # node_ids is used only in diagnostic singleton replay. Standard FARE
        # inference continues to use all N target nodes without extra gathers.
        observed_mask=inputs['obs_mask'] if node_ids is None else inputs['obs_mask'].index_select(2,node_ids)
        observed_input=inputs['x_obs'] if node_ids is None else inputs['x_obs'].index_select(2,node_ids)
        mask=observed_mask.permute(0,2,1)[:,:,None,:]
        obs=observed_input.permute(0,2,1)[:,:,None,:]
        joint=mask & hvalid
        obs_count=joint.sum(-1).clamp_min(1)
        level=((obs-hist)*joint).sum(-1)/obs_count
        level=level.clamp(-3,3)
        # Replay historical episode under current target's exact missingness.
        masked_hist=hist*joint.float()
        hist_feature=torch.cat([masked_hist,mask.expand_as(hist).float(),level[...,None],
                                hvalid.float().mean(-1,keepdim=True)],-1)
        state=self.history_encoder(hist_feature)
        # Last valid historic value must be computed via masking, not count-1 if gaps exist.
        t=torch.arange(h,device=hist.device)[None,None,None,:]
        last=(t*hvalid.long()).amax(-1)
        historic_last=torch.gather(hist,-1,last[...,None]).squeeze(-1)
        # Invalid historical targets never become evidence.
        future=future*fvalid.float()
        adjusted_future=future+level[...,None]
        # Candidate-specific future compared with base forecast; historic future is
        # strictly prior to query issue time (enforced by coarse retrieval).
        base_values=base_forecast if node_ids is None else base_forecast.index_select(2,node_ids)
        base=base_values.permute(0,2,1)[:,:,None,:]
        analog_delta=(adjusted_future-base).clamp(-5,5)
        future_features=torch.cat([future-historic_last[...,None], analog_delta, state,
                                   latent, joint.float().mean(-1,keepdim=True),
                                   fvalid.float().mean(-1,keepdim=True)],-1)
        token=self.future_encoder(future_features)
        uncertainty=torch.sigmoid(self.reliability_head(torch.cat([token,
                         joint.float().mean(-1,keepdim=True),fvalid.float().mean(-1,keepdim=True)],-1))).squeeze(-1)
        # Explicit candidate dependence; no pure Query-to-output shortcut.
        delta=(.15*analog_delta + self.delta_head(token)) * uncertainty[...,None] * fvalid.float()
        return {'token':token,'state_token':state,'state_history':self.state_head(state),
                'forecast_delta':delta,'reliability':uncertainty,
                'future_valid':fvalid,'history_valid':hvalid,
                'aligned_future':adjusted_future}


class CandidateForecastProbe(nn.Module):
    """Candidate-specific residual predictor, trained before utility labels are formed."""
    def __init__(self,token_dim=64,horizon=12):
        super().__init__()
        self.head=nn.Sequential(nn.Linear(token_dim,token_dim),nn.GELU(),nn.Linear(token_dim,horizon))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,base_forecast,evidence):
        # [B,N,K,P]; gradients flow to head in evidence-training phase.
        delta=evidence['forecast_delta']+self.head(evidence['token'])
        return base_forecast.permute(0,2,1)[:,:,None,:]+delta
