from __future__ import annotations
import torch
from torch import nn
from .stla import MaskAwareSTLAformer

class MaskRAFTSTLA(nn.Module):
    def __init__(self, stla: MaskAwareSTLAformer, periods, horizon=12, latent_gate_dim=8, fusion_hidden=64, dropout=0.1, gate_bias=-2.0):
        super().__init__(); self.stla=stla; self.periods=list(periods); self.horizon=int(horizon)
        self.period_adapters=nn.ModuleList([nn.Linear(horizon,horizon) for _ in self.periods])
        self.period_router=nn.Sequential(nn.Linear(4,32),nn.GELU(),nn.Dropout(dropout),nn.Linear(32,1))
        self.latent_gate=nn.Sequential(nn.Linear(stla.model_dim,latent_gate_dim),nn.GELU())
        fin=3+4+latent_gate_dim  # base,retr,diff + mask/conf + latent
        self.delta=nn.Sequential(nn.Linear(fin,fusion_hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(fusion_hidden,1))
        self.gate=nn.Sequential(nn.Linear(fin,fusion_hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(fusion_hidden,1))
        nn.init.constant_(self.gate[-1].bias,float(gate_bias)); nn.init.zeros_(self.delta[-1].weight); nn.init.zeros_(self.delta[-1].bias)
    def set_train_phase(self, phase, last_layers=2): self.stla.set_train_phase(phase,last_layers)
    def forward(self,x,mask,tod,dow,retrieval_evidence,retrieval_conf):
        # evidence B,G,P,N ; conf B,G,4
        base,z=self.stla(x,mask,tod,dow); B,G,P,N=retrieval_evidence.shape
        adapted=[]
        for gi,layer in enumerate(self.period_adapters): adapted.append(layer(retrieval_evidence[:,gi].transpose(1,2)).transpose(1,2))
        adapted=torch.stack(adapted,1)
        plogit=self.period_router(retrieval_conf).squeeze(-1); pw=torch.softmax(plogit,dim=1)
        retr=(adapted*pw[:,:,None,None]).sum(1)
        conf=(retrieval_conf*pw[:,:,None]).sum(1) # B,4
        node_cov=mask.mean(1); global_cov=mask.mean((1,2),keepdim=False).view(B,1).expand(B,N)
        lat=self.latent_gate(z); lat=lat[:,None,:,:].expand(B,P,N,-1)
        f=torch.cat([base.unsqueeze(-1),retr.unsqueeze(-1),(retr-base).unsqueeze(-1),node_cov[:,None,:,None].expand(B,P,N,1),global_cov[:,None,:,None].expand(B,P,N,1),conf[:,0][:,None,None,None].expand(B,P,N,1),conf[:,2][:,None,None,None].expand(B,P,N,1),lat],dim=-1)
        delta=self.delta(f).squeeze(-1); gate=torch.sigmoid(self.gate(f)).squeeze(-1); final=base+gate*delta
        return {'prediction':final,'base_prediction':base,'retrieval_prediction':retr,'gate':gate,'period_weights':pw,'confidence':conf}
