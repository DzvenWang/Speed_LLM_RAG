from __future__ import annotations
import importlib.util,sys,copy
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

def missing_age(mask):
    b,h,n=mask.shape; run=torch.zeros(b,n,device=mask.device,dtype=torch.float32);seq=[]
    for t in range(h):run=(run+1)*(~mask[:,t]).float();seq.append(run)
    return torch.stack(seq,1)/max(1,h)

class FARESTLA(nn.Module):
    """Retains official STLA blocks, adds trainable retrieval/graph/reliability adapters."""
    def __init__(self,base,num_nodes,adj, cfg):
        super().__init__();self.base=base;self.d=int(base.model_dim);self.n=num_nodes;self.cfg=cfg
        self.failure=nn.Sequential(nn.Linear(5,self.d),nn.GELU(),nn.Linear(self.d,self.d))
        self.decomp=nn.Sequential(nn.Linear(self.d+2,self.d),nn.GELU(),nn.Linear(self.d,self.d))
        self.graph_adapter=nn.ModuleDict({str(i):nn.Sequential(nn.Linear(self.d+1,self.d),nn.GELU(),nn.Linear(self.d,self.d)) for i in cfg.get('injection_layers',[4,5])})
        self.cross_adapter=nn.ModuleDict({str(i):nn.Sequential(nn.Linear(self.d*2+1,self.d),nn.GELU(),nn.Linear(self.d,self.d)) for i in cfg.get('injection_layers',[4,5])})
        self.retrieval_cross=nn.ModuleDict({str(i):nn.MultiheadAttention(self.d,4,batch_first=True) for i in cfg.get('injection_layers',[4,5])})
        self.retrieval_cross_out=nn.ModuleDict({str(i):nn.Linear(self.d,self.d) for i in cfg.get('injection_layers',[4,5])})
        self.gate=nn.Sequential(nn.Linear(self.d+3,self.d),nn.GELU(),nn.Linear(self.d,1))
        for mod in [self.failure,self.decomp,*self.graph_adapter.values(),*self.cross_adapter.values()]:
            nn.init.zeros_(mod[-1].weight);nn.init.zeros_(mod[-1].bias)
        nn.init.constant_(self.gate[-1].bias,-2.)
        for mod in self.retrieval_cross_out.values():nn.init.zeros_(mod.weight);nn.init.zeros_(mod.bias)
        a=adj.float().abs();a=a+a.T;a.fill_diagonal_(1)
        a=a/a.sum(-1,keepdim=True).clamp_min(1)
        self.register_buffer('graph_matrix',a)
    def set_phase(self,phase):
        for p in self.base.parameters():p.requires_grad=phase=='full'
        if phase in ('partial','calibration'):
            for module in list(self.base.attn_layers_st)[-2:]+[self.base.output_proj]:
                for p in module.parameters():p.requires_grad=True
        if phase=='calibration':
            for p in self.base.parameters(): p.requires_grad=False
        for n,p in self.named_parameters():
            if not n.startswith('base.'):p.requires_grad=True
        if phase not in ('partial','full'):
            self.base.eval()
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
    def forward(self,x,mask,tod,dow,state=None,innovation=None,confidence=None,retrieved_tokens=None,support_valid=None,enabled=False,return_hidden=False):
        # With enabled=False and zero adapters, this is numerically the pretrained STLA.
        b,h,n=x.shape
        if n!=self.n:raise ValueError(f'Node count mismatch {n}!={self.n}')
        if enabled and state is not None:
            recovered=state['recovered']; gx=state['gate']
            x=torch.where(mask,x,gx*recovered)
        z=self._embed(x,tod,dow)
        if enabled:
            node_cov=mask.float().mean(1,keepdim=True).expand_as(x)
            time_cov=mask.float().mean(2,keepdim=True).expand_as(x)
            unc=torch.zeros_like(x) if state is None else state['uncertainty']
            features=torch.stack([mask.float(),node_cov,time_cov,missing_age(mask),unc],-1)
            z=z+self.failure(features)
        ne=self.base.node_position_encoding(z);te=self.base.time_position_encoding(z)
        trend=self.base.MLP(z)
        if enabled:
            c=torch.zeros_like(x) if confidence is None else confidence
            trend=trend+self.decomp(torch.cat([z,c[...,None],mask[...,None].float()],-1))
        z=z-trend; pre_injection=None
        for i,block in enumerate(self.base.attn_layers_st):
            if i==min(self.cfg.get('injection_layers',[4,5])):pre_injection=z
            z=block(z,ne,te)
            if enabled and str(i) in self.graph_adapter:
                c=torch.ones_like(x) if confidence is None else confidence
                graph=torch.einsum('nm,btmd->btnd',self.graph_matrix,z*c[...,None])
                z=z+self.graph_adapter[str(i)](torch.cat([graph,c[...,None]],-1))
                if innovation is not None:
                    z=z+self.cross_adapter[str(i)](torch.cat([z,innovation,c[...,None]],-1))
                if retrieved_tokens is not None:
                    # Per-node Q = current STLA history, KV = externally retrieved Top-K.
                    # Fully invalid candidate sets are handled without all-masked Softmax.
                    k=retrieved_tokens.shape[-2]
                    q=z.permute(0,2,1,3).reshape(b*n,h,self.d)
                    kv=retrieved_tokens.reshape(b*n,k,self.d)
                    if support_valid is not None:
                        vm=support_valid.reshape(b*n,k).clone()
                        all_bad=~vm.any(-1);vm[all_bad,0]=True
                        padding=~vm
                    else:
                        padding=None;all_bad=torch.zeros(b*n,dtype=torch.bool,device=z.device)
                    ret,_=self.retrieval_cross[str(i)](q,kv,kv,key_padding_mask=padding,need_weights=False)
                    ret=ret.masked_fill(all_bad[:,None,None],0).reshape(b,n,h,self.d).permute(0,2,1,3)
                    z=z+self.retrieval_cross_out[str(i)](ret)*c[...,None]
        hfinal=z+trend
        pred=self.base.output_proj(hfinal.transpose(1,2).reshape(b,n,h*self.d)).view(b,n,self.base.out_steps,self.base.output_dim).transpose(1,2).squeeze(-1)
        if pre_injection is None:pre_injection=z
        return {'prediction':pred,'hidden':pre_injection,'final_hidden':hfinal} if return_hidden else {'prediction':pred,'hidden':pre_injection}
