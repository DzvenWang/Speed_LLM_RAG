from __future__ import annotations
import importlib.util,sys
from pathlib import Path
import torch
from torch import nn
from .config import project_root

def load_pretrained_stla(path,device='cpu'):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('model')!='STLAformer': raise ValueError('Only STLAformer checkpoints are supported')
    src=project_root()/'Backbone'/'vendor'/'STLAformer'/'model'/'STLformer.py'
    if not src.is_file(): raise FileNotFoundError(f'STLAformer vendor source not installed: {src}. Run Backbone setup_official_models.sh on your machine.')
    if str(src.parent) not in sys.path:sys.path.insert(0,str(src.parent))
    spec=importlib.util.spec_from_file_location('fare_stla_vendor',src); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mc=ck['model_config'];kw=dict(num_nodes=int(ck['num_nodes']),in_steps=12,out_steps=12,steps_per_week=int(mc.get('steps_per_week',2016)),input_dim=int(mc.get('input_dim',2)),output_dim=1,input_embedding_dim=int(mc.get('input_embedding_dim',32)),tow_embedding_dim=int(mc.get('tow_embedding_dim',32)),feed_forward_dim=int(mc.get('feed_forward_dim',256)),num_heads=int(mc.get('num_heads',4)),num_layers=int(mc.get('num_layers',6)),dropout=float(mc.get('dropout',.3)))
    m=mod.STLformer(**kw);m.load_state_dict(ck['state_dict'],strict=True)
    return m.to(device),ck

class FARESTLA(nn.Module):
    """Frozen STLAformer with retrieval-only cross-attention at selected layers.

    Unlike legacy FARE there are no independent graph/failure/decomposition
    prediction adapters. Without a valid external token, predictions equal the
    frozen STLAformer exactly (including after training cross-attention bias).
    """
    def __init__(self,base,num_nodes,adj,cfg):
        super().__init__()
        self.base=base
        self.d=int(base.model_dim)
        self.n=int(num_nodes)
        layers=cfg.get('injection_layers',[4,5])
        self.injection_layers=tuple(int(v) for v in layers)
        if not self.injection_layers:
            raise ValueError('At least one retrieval injection layer is required')
        if min(self.injection_layers)<0 or max(self.injection_layers)>=len(base.attn_layers_st):
            raise ValueError('Retrieval injection layer outside STLAformer')
        self.retrieval_cross=nn.ModuleDict({str(i):nn.MultiheadAttention(self.d,4,batch_first=True)
                                          for i in self.injection_layers})
        self.retrieval_cross_out=nn.ModuleDict({str(i):nn.Linear(self.d,self.d,bias=False)
                                              for i in self.injection_layers})
        for mod in self.retrieval_cross_out.values(): nn.init.zeros_(mod.weight)
        for p in self.base.parameters():p.requires_grad=False
        self.base.eval()

    def train(self,mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def _embed(self,x,tod,dow):
        b,h,n=x.shape
        slot=dow.long()*288+tod.long()
        frac=slot.float()/float(self.base.steps_per_week)
        frac=torch.nextafter(frac,torch.full_like(frac,float('inf')))
        tg=frac[...,None].expand(b,h,n)
        raw=torch.stack([x,tg],-1)
        feat=[self.base.input_proj(raw[...,:self.base.input_dim])]
        if self.base.tow_embedding_dim>0:
            feat.append(self.base.tow_embedding((tg*self.base.steps_per_week).long()))
        return torch.cat(feat,-1)

    def forward(self,x,mask,tod,dow,enabled=False,return_hidden=False,
                retrieved_tokens=None,support_valid=None):
        b,h,n=x.shape
        if n!=self.n:raise ValueError(f'STLA node count mismatch: {n}!={self.n}')
        z=self._embed(x,tod,dow)
        ne=self.base.node_position_encoding(z)
        te=self.base.time_position_encoding(z)
        trend=self.base.MLP(z)
        z=z-trend
        pre=None
        for i,block in enumerate(self.base.attn_layers_st):
            if i==min(self.injection_layers):pre=z
            z=block(z,ne,te)
            if enabled and retrieved_tokens is not None and str(i) in self.retrieval_cross:
                k=retrieved_tokens.shape[-2]
                q=z.permute(0,2,1,3).reshape(b*n,h,self.d)
                kv=retrieved_tokens.reshape(b*n,k,self.d)
                if support_valid is None:
                    any_valid=torch.ones(b*n,device=z.device,dtype=torch.bool)
                    padding=None
                else:
                    vv=support_valid.reshape(b*n,k).clone()
                    any_valid=vv.any(-1)
                    vv[~any_valid,0]=True
                    padding=~vv
                attention,_=self.retrieval_cross[str(i)](q,kv,kv,
                                         key_padding_mask=padding,need_weights=False)
                attention=self.retrieval_cross_out[str(i)](attention)
                attention=attention*any_valid[:,None,None].to(attention.dtype)
                z=z+attention.reshape(b,n,h,self.d).permute(0,2,1,3)
        final=z+trend
        pred=self.base.output_proj(final.transpose(1,2).reshape(b,n,h*self.d))
        pred=pred.view(b,n,self.base.out_steps,self.base.output_dim).transpose(1,2).squeeze(-1)
        if pre is None:pre=z
        return {'prediction':pred,'hidden':pre,'final_hidden':final} if return_hidden else {'prediction':pred,'hidden':pre}
