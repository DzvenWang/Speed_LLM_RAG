from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import torch
from torch import nn


def _load_module(path: Path):
    model_dir=path.parent
    if str(model_dir) not in sys.path: sys.path.insert(0,str(model_dir))
    spec=importlib.util.spec_from_file_location('tcrr_stla_vendor_stla',path)
    if spec is None or spec.loader is None: raise RuntimeError(f'Cannot import {path}')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def load_pretrained_stla(project_root: Path, checkpoint: Path, device: torch.device):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if str(ck.get('model'))!='STLAformer': raise RuntimeError(f'Checkpoint is not STLAformer: {checkpoint}')
    mc=dict(ck['model_config']); n=int(ck['num_nodes'])
    src=project_root/'Backbone'/'vendor'/'STLAformer'/'model'/'STLformer.py'
    if not src.is_file(): raise FileNotFoundError(f'Missing official STLAformer source: {src}')
    mod=_load_module(src)
    base=mod.STLformer(num_nodes=n,in_steps=12,out_steps=12,steps_per_week=int(mc.get('steps_per_week',2016)),input_dim=int(mc.get('input_dim',2)),output_dim=1,input_embedding_dim=int(mc.get('input_embedding_dim',32)),tow_embedding_dim=int(mc.get('tow_embedding_dim',32)),feed_forward_dim=int(mc.get('feed_forward_dim',256)),num_heads=int(mc.get('num_heads',4)),num_layers=int(mc.get('num_layers',6)),dropout=float(mc.get('dropout',0.3)))
    base.load_state_dict(ck['state_dict'],strict=True); base.to(device)
    return base,ck

def missing_age(mask: torch.Tensor) -> torch.Tensor:
    B,H,N=mask.shape; run=torch.zeros(B,N,device=mask.device,dtype=mask.dtype); out=[]
    for t in range(H):
        run=(run+1.0)*(1.0-mask[:,t]); out.append(run)
    return torch.stack(out,dim=1)/max(H,1)

class MaskAwareSTLAformer(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__(); self.base=base; d=int(base.model_dim)
        self.mask_proj=nn.Sequential(nn.Linear(4,d),nn.GELU(),nn.Linear(d,d))
        nn.init.zeros_(self.mask_proj[-1].weight); nn.init.zeros_(self.mask_proj[-1].bias)
    @property
    def model_dim(self): return int(self.base.model_dim)
    def _tow(self,tod,dow):
        slot=dow.long()*288+tod.long(); frac=slot.float()/float(self.base.steps_per_week)
        return torch.nextafter(frac,torch.full_like(frac,float('inf')))
    def set_train_phase(self, phase: str, last_layers: int=2):
        for p in self.base.parameters(): p.requires_grad=(phase=='full')
        if phase=='partial':
            for p in self.base.output_proj.parameters(): p.requires_grad=True
            for layer in list(self.base.attn_layers_st)[-int(last_layers):]:
                for p in layer.parameters(): p.requires_grad=True
        for p in self.mask_proj.parameters(): p.requires_grad=True
    def forward(self,x,mask,tod,dow):
        B,H,N=x.shape; tow=self._tow(tod,dow); tow_grid=tow.unsqueeze(-1).expand(B,H,N); raw=torch.stack([x,tow_grid],dim=-1)
        z_in=self.base.input_proj(raw[...,:self.base.input_dim]); features=[z_in]
        if self.base.tow_embedding_dim>0: features.append(self.base.tow_embedding((tow_grid*self.base.steps_per_week).long()))
        z=torch.cat(features,dim=-1)
        nc=mask.mean(dim=1,keepdim=True).expand(-1,H,-1); tc=mask.mean(dim=2,keepdim=True).expand(-1,-1,N); age=missing_age(mask)
        mfeat=torch.stack([mask,nc,tc,age],dim=-1); z=z+self.mask_proj(mfeat)
        node_emb=self.base.node_position_encoding(z); time_emb=self.base.time_position_encoding(z)
        bx=self.base.MLP(z); h=z-bx
        for attn in self.base.attn_layers_st: h=attn(h,node_emb,time_emb)
        h=h+bx
        out=h.transpose(1,2).reshape(B,self.base.num_nodes,self.base.in_steps*self.base.model_dim)
        out=self.base.output_proj(out).view(B,self.base.num_nodes,self.base.out_steps,self.base.output_dim).transpose(1,2)
        return out.squeeze(-1), h.mean(dim=1)
