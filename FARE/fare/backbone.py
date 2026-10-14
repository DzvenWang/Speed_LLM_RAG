"""Frozen official STLAformer loader and optional differentiable evidence injection.

No pretrained weights are bundled. Uses the installed Backbone official vendor source.
"""
from pathlib import Path
import importlib.util,sys
import torch
from torch import nn
from .config import root

class FrozenSTLA(nn.Module):
    def __init__(self,model,ckpt,inject_layers=(4,5)):
        super().__init__();self.base=model
        self.d=int(model.model_dim);self.mean=float(ckpt['mean']);self.std=float(ckpt['std'])
        self.n=int(ckpt['num_nodes'])
        self.layers=tuple(i for i in inject_layers if 0<=i<len(model.attn_layers_st))
        self.inject=nn.ModuleDict({str(i):nn.MultiheadAttention(self.d,4,batch_first=True) for i in self.layers})
        self.project=nn.ModuleDict({str(i):nn.Linear(self.d,self.d,bias=False) for i in self.layers})
        for v in self.project.values():nn.init.zeros_(v.weight)
        for p in model.parameters():p.requires_grad_(False)
        model.eval()
    def train(self,mode=True):
        super().train(mode);self.base.eval();return self
    def _embed(self,x,tod,dow):
        b,h,n=x.shape
        slot=dow.long()*288+tod.long()
        frac=torch.nextafter(slot.float()/float(self.base.steps_per_week),torch.full_like(slot.float(),float('inf')))
        tg=frac[...,None].expand(b,h,n)
        z=torch.stack([x,tg],-1)
        part=[self.base.input_proj(z[...,:self.base.input_dim])]
        if self.base.tow_embedding_dim>0:part.append(self.base.tow_embedding((tg*self.base.steps_per_week).long()))
        return torch.cat(part,-1)
    def forward(self,x,mask,tod,dow,tokens=None,token_valid=None):
        b,h,n=x.shape
        if n!=self.n:raise ValueError('Checkpoint node count mismatch')
        def _core():
            z=self._embed(x,tod,dow)
            ne=self.base.node_position_encoding(z);te=self.base.time_position_encoding(z)
            trend=self.base.MLP(z);z=z-trend
            hidden=None
            for i,block in enumerate(self.base.attn_layers_st):
                if hidden is None and i>=min(self.layers or (0,)):hidden=z.mean(1)
                z=block(z,ne,te)
                if tokens is not None and str(i) in self.inject:
                    k=tokens.shape[2]
                    query=z.permute(0,2,1,3).reshape(b*n,h,self.d)
                    kv=tokens.reshape(b*n,k,self.d)
                    valid=token_valid.reshape(b*n,k).clone() if token_valid is not None else torch.ones(b*n,k,dtype=torch.bool,device=z.device)
                    active=valid.any(-1);valid[~active,0]=True
                    att,_=self.inject[str(i)](query,kv,kv,key_padding_mask=~valid,need_weights=False)
                    z=z+self.project[str(i)](att).reshape(b,n,h,self.d).permute(0,2,1,3)*active.reshape(b,1,n,1).to(z.dtype)
            final=z+trend
            pred=self.base.output_proj(final.transpose(1,2).reshape(b,n,h*self.d))
            pred=pred.view(b,n,self.base.out_steps,self.base.output_dim).transpose(1,2).squeeze(-1)
            return {'prediction':pred,'hidden':hidden if hidden is not None else z.mean(1)}
        if tokens is None:
            with torch.no_grad():return _core()
        return _core()

def load_stla(checkpoint,device='cpu',inject_layers=(4,5)):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck.get('model')!='STLAformer':raise ValueError('Checkpoint is not an STLAformer model')
    src=root()/'Backbone'/'vendor'/'STLAformer'/'model'/'STLformer.py'
    if not src.is_file():raise FileNotFoundError(f'Missing official vendor STLAformer source: {src}')
    if str(src.parent) not in sys.path:sys.path.insert(0,str(src.parent))
    spec=importlib.util.spec_from_file_location('fare3_stla_vendor',src);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mc=ck['model_config']
    kwargs=dict(num_nodes=int(ck['num_nodes']),in_steps=12,out_steps=12,
                steps_per_week=int(mc.get('steps_per_week',2016)),input_dim=int(mc.get('input_dim',2)),
                output_dim=1,input_embedding_dim=int(mc.get('input_embedding_dim',32)),tow_embedding_dim=int(mc.get('tow_embedding_dim',32)),
                feed_forward_dim=int(mc.get('feed_forward_dim',256)),num_heads=int(mc.get('num_heads',4)),
                num_layers=int(mc.get('num_layers',6)),dropout=float(mc.get('dropout',.3)))
    model=mod.STLformer(**kwargs);model.load_state_dict(ck['state_dict'],strict=True)
    return FrozenSTLA(model,ck,inject_layers).to(device)
