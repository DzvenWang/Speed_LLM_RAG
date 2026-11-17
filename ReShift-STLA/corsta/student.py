"""Trainable STLAformer with a shared prefix and late-block refinement.

The official vendor model is loaded by corsta.stla.load_pretrained_stla. Its
input/position/residual/head code is reproduced in named steps and covered by
an equivalence test on an actual vendor checkpoint in preflight.
"""
from __future__ import annotations
import torch
from .stla import MaskAwareSTLAformer, missing_age


class SplitSTLA(MaskAwareSTLAformer):
    def __init__(self, base, tail_layers=2):
        super().__init__(base)
        count=len(self.base.attn_layers_st)
        if not 1 <= tail_layers < count:
            raise ValueError(f'tail_layers must be in 1..{count-1}')
        self.tail_layers=int(tail_layers)

    def encode_prefix(self, x, mask, tod, dow):
        if x.shape!=mask.shape or x.ndim!=3: raise ValueError('history/mask mismatch')
        B,H,N=x.shape
        if H!=self.base.in_steps or N!=self.base.num_nodes:
            raise ValueError('STLA history or node count mismatch')
        mask=mask.float().clamp(0,1)
        x=torch.where(mask.bool(),torch.nan_to_num(x.float()),torch.zeros_like(x.float()))
        tow=self._tow(tod,dow)
        tow_grid=tow.unsqueeze(-1).expand(B,H,N)
        raw=torch.stack((x,tow_grid),dim=-1)
        input_feat=self.base.input_proj(raw[...,:self.base.input_dim])
        features=[input_feat]
        if self.base.tow_embedding_dim>0:
            features.append(self.base.tow_embedding((tow_grid*self.base.steps_per_week).long()))
        z=torch.cat(features,-1)
        nc=mask.mean(1,keepdim=True).expand(-1,H,-1)
        tc=mask.mean(2,keepdim=True).expand(-1,-1,N)
        age=missing_age(mask)
        z=z+self.mask_proj(torch.stack((mask,nc,tc,age),-1))
        node_emb=self.base.node_position_encoding(z)
        time_emb=self.base.time_position_encoding(z)
        bx=self.base.MLP(z)
        hidden=z-bx
        for layer in list(self.base.attn_layers_st)[:-self.tail_layers]:
            hidden=layer(hidden,node_emb,time_emb)
        return hidden, (node_emb,time_emb,bx)

    def encode_tail(self, hidden, context):
        node_emb,time_emb,bx=context
        for layer in list(self.base.attn_layers_st)[-self.tail_layers:]:
            hidden=layer(hidden,node_emb,time_emb)
        return hidden+bx

    def forecast_head(self, features):
        B,H,N,D=features.shape
        if H!=self.base.in_steps or N!=self.base.num_nodes:raise ValueError('head shape mismatch')
        out=features.transpose(1,2).reshape(B,N,H*D)
        out=self.base.output_proj(out).view(B,N,self.base.out_steps,self.base.output_dim).transpose(1,2)
        if self.base.output_dim!=1:raise ValueError('CoRSTA expects output_dim=1')
        return out.squeeze(-1)

    def forward_features(self,x,mask,tod,dow):
        prefix,ctx=self.encode_prefix(x,mask,tod,dow)
        h=self.encode_tail(prefix,ctx)
        return prefix,h,self.forecast_head(h),ctx

    def forward(self,x,mask,tod,dow):
        _,h,p,_=self.forward_features(x,mask,tod,dow)
        return p,h.mean(1)
