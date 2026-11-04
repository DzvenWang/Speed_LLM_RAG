"""C1: observable-error anchored structured counterfactual response.

The normal branch does not depend on CF. The CF utility head affects the
actual CF correction, and correct/shuffled predictions can be supervised
without putting future labels into the forward pass.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class TemporalResponseEncoder(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(2,hidden,3,padding=1,bias=False),nn.GELU(),
            nn.Conv1d(hidden,hidden,3,padding=2,dilation=2,bias=False),nn.GELU(),
            nn.Conv1d(hidden,1,1,bias=False))

    def forward(self, normal, cf):
        B,K,P,N=cf.shape
        z=torch.stack((normal,cf),2).permute(0,1,4,2,3).reshape(B*K*N,2,P)
        return self.net(z).reshape(B,K,N,P).permute(0,1,3,2)

class StructuredCounterfactual(nn.Module):
    def __init__(self,horizon,latent,hidden=64,temporal_hidden=16):
        super().__init__()
        self.horizon_embed=nn.Embedding(horizon,8)
        common_dim=latent+8+4
        self.normal_net=nn.Sequential(nn.Linear(common_dim,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.temporal=TemporalResponseEncoder(temporal_hidden)
        self.cf_net=nn.Sequential(nn.Linear(common_dim+5,hidden),nn.GELU(),
                                  nn.Linear(hidden,hidden),nn.GELU())
        self.amplitude=nn.Linear(hidden,1)
        self.gate=nn.Linear(hidden,1)
        self.incremental_utility=nn.Linear(hidden,1)
        self.gain_probability=nn.Linear(hidden,1)
        nn.init.zeros_(self.normal_net[-1].weight);nn.init.zeros_(self.normal_net[-1].bias)
        for head in (self.amplitude,self.incremental_utility,self.gain_probability):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)

    def _cf_increment(self, normal, cf, common):
        temporal=self.temporal(normal,cf)
        pair=torch.stack((cf,temporal,normal*cf,cf.abs(),(normal-cf).abs()),-1)
        feature=self.cf_net(torch.cat((common,pair),-1))
        amplitude=1+.75*torch.tanh(self.amplitude(feature)).squeeze(-1)
        # Transfer amplitude controller, NOT a calibrated probability.
        transfer_gate=torch.sigmoid(self.gate(feature)).squeeze(-1)
        predicted_utility=self.incremental_utility(feature).squeeze(-1)
        # Return the raw logit as well as the probability: BCEWithLogitsLoss
        # is AMP-safe; BCE(sigmoid(logit)) is forbidden by CUDA autocast.
        positive_logit=self.gain_probability(feature).squeeze(-1)
        positive_prob=torch.sigmoid(positive_logit)
        # Utility is connected to inference; initialized neutral, bounded to avoid collapse.
        utility_scale=(1.+.5*torch.tanh(predicted_utility))* (.5+positive_prob)
        increment=cf*amplitude*transfer_gate*utility_scale*(1+.25*torch.tanh(temporal))
        return increment,temporal,transfer_gate,predicted_utility,positive_prob,positive_logit

    def forward(self,normal,cf,base,context,score,coverage,enabled=True,
                shuffled_negatives=False):
        normal=normal.float();cf=cf.float()
        B,K,P,N=normal.shape
        if cf.shape!=normal.shape:raise ValueError('normal and CF shape mismatch')
        hz=self.horizon_embed(torch.arange(P,device=normal.device))[None,None,:,None].expand(B,K,P,N,8)
        lat=context[:,None,None].expand(B,K,P,N,context.shape[-1])
        extra=torch.stack((base[:,None].expand_as(normal),score[:,:,None,None].expand_as(normal),
                           coverage[:,:,None,None].expand_as(normal),torch.tanh(normal)),-1)
        common=torch.cat((lat.float(),hz.float(),extra.float()),-1)
        normal_scale=1+.5*torch.tanh(self.normal_net(common)).squeeze(-1)
        normal_component=normal*normal_scale
        zeros=torch.zeros_like(normal)
        if enabled:
            increment,temporal,gate,utility,prob,prob_logit=self._cf_increment(normal,cf,common)
        else:
            increment=temporal=gate=utility=prob=prob_logit=zeros
        out=dict(normal_component=normal_component,cf_increment=increment,
                 cf_response=cf,cf_temporal=temporal,cf_reliability=gate,
                 cf_transfer_gate=gate,
                 cf_incremental_utility=utility,cf_gain_probability=prob,
                 cf_gain_logit=prob_logit,
                 candidate_correction=normal_component+increment)
        if enabled and shuffled_negatives and K>1:
            # Diagnostics/training: pair the same historical normal residual with
            # another candidate's CF response. No current-future labels here.
            wrong_cf=cf.roll(1,dims=1)
            neg_inc,_,_,neg_util,_,_=self._cf_increment(normal,wrong_cf,common)
            out['cf_wrong_increment']=neg_inc
            out['cf_wrong_utility_pred']=neg_util
        return out
