"""Frozen STLAformer with reusable pre-injection activations.

Parameters are frozen but evidence-bearing suffix executes with autograd enabled
so adapter tokens and attention weights receive correct forecast gradients.
"""
from pathlib import Path
import sys,importlib.util
import torch
from torch import nn
from .config import root

class FrozenSTLA(nn.Module):
    def __init__(self,model,ckpt,inject_layers=(4,5)):
        super().__init__();self.base=model
        self.d=int(model.model_dim);self.n=int(ckpt['num_nodes'])
        self.mean=float(ckpt['mean']);self.std=float(ckpt['std'])
        self.layers=tuple(int(i) for i in inject_layers if 0<=i<len(model.attn_layers_st))
        self.split=min(self.layers) if self.layers else len(model.attn_layers_st)
        self.inject=nn.ModuleDict({str(i):nn.MultiheadAttention(self.d,4,batch_first=True)
                                   for i in self.layers})
        self.project=nn.ModuleDict({str(i):nn.Linear(self.d,self.d,bias=False)
                                    for i in self.layers})
        for p in self.project.values():nn.init.zeros_(p.weight)
        for p in model.parameters():p.requires_grad_(False)
        model.eval()
    def train(self,mode=True):
        super().train(mode);self.base.eval();return self
    def _embed(self,x,tod,dow):
        b,h,n=x.shape
        slot=dow.long()*288+tod.long()
        frac=torch.nextafter(slot.float()/self.base.steps_per_week,
                             torch.full_like(slot.float(),float('inf')))
        time=frac[...,None].expand(b,h,n)
        z=torch.stack([x,time],-1)
        parts=[self.base.input_proj(z[...,:self.base.input_dim])]
        if self.base.tow_embedding_dim>0:
            parts.append(self.base.tow_embedding((time*self.base.steps_per_week).long()))
        return torch.cat(parts,-1)
    def encode_prefix(self,x,mask,tod,dow):
        b,h,n=x.shape
        if n!=self.n:raise ValueError('STLAformer checkpoint node count mismatch')
        with torch.no_grad():
            z=self._embed(x,tod,dow)
            ne=self.base.node_position_encoding(z);te=self.base.time_position_encoding(z)
            trend=self.base.MLP(z)
            z=z-trend
            for i in range(self.split):z=self.base.attn_layers_st[i](z,ne,te)
            hidden=z.mean(1)
        return {'z':z,'ne':ne,'te':te,'trend':trend,'hidden':hidden}
    def _decode(self,cache,tokens=None,token_valid=None):
        z=cache['z'];b,h,n,d=z.shape
        for i in range(self.split,len(self.base.attn_layers_st)):
            z=self.base.attn_layers_st[i](z,cache['ne'],cache['te'])
            if tokens is not None and str(i) in self.inject:
                k=tokens.shape[2]
                q=z.permute(0,2,1,3).reshape(b*n,h,d)
                kv=tokens.reshape(b*n,k,d)
                good=(token_valid.reshape(b*n,k).clone() if token_valid is not None else
                      torch.ones(b*n,k,device=z.device,dtype=torch.bool))
                active=good.any(-1)
                good[~active,0]=True
                att,_=self.inject[str(i)](q,kv,kv,key_padding_mask=~good,need_weights=False)
                dz=self.project[str(i)](att).view(b,n,h,d).permute(0,2,1,3)
                z=z+dz*active.reshape(b,1,n,1).to(z.dtype)
        out=z+cache['trend']
        pred=self.base.output_proj(out.transpose(1,2).reshape(b,n,h*d))
        pred=pred.reshape(b,n,self.base.out_steps,self.base.output_dim).transpose(1,2).squeeze(-1)
        return pred
    def encode_base(self,x,mask,tod,dow):
        cache=self.encode_prefix(x,mask,tod,dow)
        with torch.no_grad():pred=self._decode(cache)
        return {'prediction':pred,'hidden':cache['hidden'],'prefix_cache':cache}
    def decode_evidence(self,cache,tokens,token_valid):
        return self._decode(cache,tokens,token_valid)
    def forward(self,x,mask,tod,dow,tokens=None,token_valid=None):
        base=self.encode_base(x,mask,tod,dow)
        if tokens is not None:
            pred=self.decode_evidence(base['prefix_cache'],tokens,token_valid)
        else:pred=base['prediction']
        return {'prediction':pred,'hidden':base['hidden']}


def load_stla(checkpoint,device='cpu',inject_layers=(4,5)):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck.get('model')!='STLAformer':raise ValueError('Checkpoint is not STLAformer')
    src=root()/'Backbone'/'vendor'/'STLAformer'/'model'/'STLformer.py'
    if not src.is_file():raise FileNotFoundError(f'Missing STLAformer vendor source: {src}')
    if str(src.parent) not in sys.path:sys.path.insert(0,str(src.parent))
    spec=importlib.util.spec_from_file_location('fare4_stla_vendor',src)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mc=ck['model_config']
    kwargs=dict(num_nodes=int(ck['num_nodes']),in_steps=12,out_steps=12,
                steps_per_week=int(mc.get('steps_per_week',2016)),input_dim=int(mc.get('input_dim',2)),
                output_dim=1,input_embedding_dim=int(mc.get('input_embedding_dim',32)),
                tow_embedding_dim=int(mc.get('tow_embedding_dim',32)),
                feed_forward_dim=int(mc.get('feed_forward_dim',256)),num_heads=int(mc.get('num_heads',4)),
                num_layers=int(mc.get('num_layers',6)),dropout=float(mc.get('dropout',.3)))
    model=mod.STLformer(**kwargs);model.load_state_dict(ck['state_dict'],strict=True)
    return FrozenSTLA(model,ck,inject_layers).to(device)
