from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import torch
from torch import nn

def _load_module(path: Path):
    if str(path.parent) not in sys.path: sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('mask_rast_stla_vendor_stla',path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def load_pretrained_stla(project_root: Path, checkpoint: Path, device):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False); mc=dict(ck['model_config']); src=project_root/'Backbone'/'vendor'/'STLAformer'/'model'/'STLformer.py'
    if str(ck.get('model'))!='STLAformer': raise RuntimeError('Backbone checkpoint is not STLAformer')
    if not src.is_file(): raise FileNotFoundError(src)
    mod=_load_module(src); b=mod.STLformer(num_nodes=int(ck['num_nodes']),in_steps=12,out_steps=12,steps_per_week=int(mc.get('steps_per_week',2016)),input_dim=int(mc.get('input_dim',2)),output_dim=1,input_embedding_dim=int(mc.get('input_embedding_dim',32)),tow_embedding_dim=int(mc.get('tow_embedding_dim',32)),feed_forward_dim=int(mc.get('feed_forward_dim',256)),num_heads=int(mc.get('num_heads',4)),num_layers=int(mc.get('num_layers',6)),dropout=float(mc.get('dropout',0.3))); b.load_state_dict(ck['state_dict'],strict=True); return b.to(device),ck

def missing_age(mask):
    B,H,N=mask.shape; r=torch.zeros(B,N,device=mask.device,dtype=mask.dtype); out=[]
    for t in range(H): r=(r+1)*(1-mask[:,t]);out.append(r)
    return torch.stack(out,1)/max(H,1)

class MaskSupportSTLAformer(nn.Module):
    def __init__(self,base,support_dim,latent_gate_bias=-1.0):
        super().__init__(); self.base=base; d=int(base.model_dim); self.mask_proj=nn.Sequential(nn.Linear(4,d),nn.GELU(),nn.Linear(d,d)); self.support_proj=nn.Linear(support_dim,d); self.support_gate=nn.Sequential(nn.Linear(support_dim+2,32),nn.GELU(),nn.Linear(32,1))
        nn.init.zeros_(self.mask_proj[-1].weight);nn.init.zeros_(self.mask_proj[-1].bias);nn.init.zeros_(self.support_proj.weight);nn.init.zeros_(self.support_proj.bias);nn.init.constant_(self.support_gate[-1].bias,float(latent_gate_bias))
    @property
    def model_dim(self): return int(self.base.model_dim)
    def set_train_phase(self,phase,last_layers=2):
        for p in self.base.parameters(): p.requires_grad=(phase=='full')
        if phase=='partial':
            for p in self.base.output_proj.parameters(): p.requires_grad=True
            for layer in list(self.base.attn_layers_st)[-int(last_layers):]:
                for p in layer.parameters(): p.requires_grad=True
        for module in (self.mask_proj,self.support_proj,self.support_gate):
            for p in module.parameters(): p.requires_grad=True
    def _tow(self,tod,dow):
        slot=dow.long()*288+tod.long(); f=slot.float()/float(self.base.steps_per_week); return torch.nextafter(f,torch.full_like(f,float('inf')))
    def forward(self,x,mask,tod,dow,support=None,reliability=None):
        B,H,N=x.shape;tow=self._tow(tod,dow);tow_grid=tow.unsqueeze(-1).expand(B,H,N);raw=torch.stack([x,tow_grid],-1); zin=self.base.input_proj(raw[...,:self.base.input_dim]); feat=[zin]
        if self.base.tow_embedding_dim>0: feat.append(self.base.tow_embedding((tow_grid*self.base.steps_per_week).long()))
        z=torch.cat(feat,-1);nc=mask.mean(1,keepdim=True).expand(-1,H,-1);tc=mask.mean(2,keepdim=True).expand(-1,-1,N);age=missing_age(mask);z=z+self.mask_proj(torch.stack([mask,nc,tc,age],-1))
        latent_gate=None
        if support is not None:
            rel=reliability if reliability is not None else torch.ones(B,N,1,device=x.device,dtype=x.dtype); gf=torch.cat([support,rel,nc[:,0,:,None]],-1); latent_gate=torch.sigmoid(self.support_gate(gf)); z=z+latent_gate[:,None,:,:]*self.support_proj(support)[:,None,:,:]
        node_emb=self.base.node_position_encoding(z);time_emb=self.base.time_position_encoding(z);bx=self.base.MLP(z);h=z-bx
        for a in self.base.attn_layers_st:h=a(h,node_emb,time_emb)
        h=h+bx;out=h.transpose(1,2).reshape(B,self.base.num_nodes,self.base.in_steps*self.base.model_dim);out=self.base.output_proj(out).view(B,self.base.num_nodes,self.base.out_steps,self.base.output_dim).transpose(1,2).squeeze(-1)
        return out,h.mean(1),latent_gate
