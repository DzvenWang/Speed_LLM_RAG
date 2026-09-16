from __future__ import annotations
import torch
from torch import nn


class NodeSensingEncoder(nn.Module):
    def __init__(self,din,N,hidden=160,demb=64,node_dim=24,dropout=0.1):
        super().__init__(); self.N=int(N); self.demb=int(demb); self.node_dim=int(node_dim)
        self.node_emb=nn.Embedding(self.N,self.node_dim)
        self.net=nn.Sequential(
            nn.Linear(int(din)+self.node_dim,int(hidden)),nn.LayerNorm(int(hidden)),nn.GELU(),nn.Dropout(float(dropout)),
            nn.Linear(int(hidden),self.demb),nn.LayerNorm(self.demb),nn.GELU(),
        )
    def forward(self,x):
        # x [B,N,F]
        B,N,_=x.shape
        if N!=self.N: raise ValueError(f'Node count mismatch {N} != {self.N}')
        e=self.node_emb.weight[None,:,:].expand(B,-1,-1)
        return self.net(torch.cat([x,e],-1))


class NodeM1Model(nn.Module):
    def __init__(self,din,N,P,hidden=160,demb=64,node_dim=24,dropout=0.1):
        super().__init__(); self.P=int(P); self.encoder=NodeSensingEncoder(din,N,hidden,demb,node_dim,dropout); self.head=nn.Linear(int(demb),self.P)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self,x):
        z=self.encoder(x); corr=self.head(z).permute(0,2,1).contiguous(); return corr,z


class NodeUsefulnessNet(nn.Module):
    """Outcome-grounded node-conditioned support shortlist scorer.

    v0.7 keeps M3 deliberately narrow: it is the high-recall node-local shortlist
    generator. Its direct correction remains a diagnostic ablation; M4 is responsible
    for node-horizon compatibility and numerical composition.
    """
    def __init__(self,demb,sense_dim=6,hidden=128,dropout=0.1,max_scale=1.5):
        super().__init__(); self.demb=int(demb); self.sense_dim=int(sense_dim); self.max_scale=float(max_scale)
        din=3*self.demb + 3*self.sense_dim + 1
        self.body=nn.Sequential(nn.Linear(din,int(hidden)),nn.GELU(),nn.Dropout(float(dropout)),nn.Linear(int(hidden),96),nn.GELU())
        self.cls=nn.Linear(96,3); self.rank=nn.Linear(96,1); self.scale=nn.Linear(96,1)
    def forward(self,q,c,sim,qs,cs):
        # q [B,N,D], c [B,K,N,D], qs [B,N,S], cs [B,K,N,S], sim [B,K]
        B,K,N,_=c.shape
        qe=q[:,None,:,:].expand(-1,K,-1,-1); qse=qs[:,None,:,:].expand(-1,K,-1,-1)
        se=sim[:,:,None,None].expand(-1,-1,N,1)
        h=self.body(torch.cat([qe,c,torch.abs(qe-c),qse,cs,torch.abs(qse-cs),se],-1))
        scale=torch.sigmoid(self.scale(h).squeeze(-1))*self.max_scale
        return self.cls(h),self.rank(h).squeeze(-1),scale


class NodeHorizonRouter(nn.Module):
    """Legacy v0.6 router retained only for checkpoint/debug compatibility."""
    def __init__(self,demb,P,sense_dim=6,hidden=128,horizon_dim=12,max_scale=1.5,dropout=0.1):
        super().__init__(); self.P=int(P); self.max_scale=float(max_scale); self.hdim=int(horizon_dim)
        din=3*int(demb)+3*int(sense_dim)
        self.pair=nn.Sequential(nn.Linear(din,int(hidden)),nn.GELU(),nn.Dropout(float(dropout)),nn.Linear(int(hidden),96),nn.GELU())
        self.h_emb=nn.Embedding(self.P,self.hdim)
        self.role=nn.Sequential(nn.Linear(96+self.hdim+4,96),nn.GELU(),nn.Dropout(0.05),nn.Linear(96,2))
        self.noop=nn.Sequential(nn.Linear(int(demb)+int(sense_dim)+self.hdim+1,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,q,c,qs,cs,rr_norm,base_norm,m3_conf):
        B,N,L,_=c.shape; qe=q[:,:,None,:].expand(-1,-1,L,-1); qse=qs[:,:,None,:].expand(-1,-1,L,-1)
        pair=self.pair(torch.cat([qe,c,torch.abs(qe-c),qse,cs,torch.abs(qse-cs)],-1))
        pe=pair[:,:,:,None,:].expand(-1,-1,-1,self.P,-1)
        he=self.h_emb.weight[None,None,None,:,:].expand(B,N,L,-1,-1)
        be=base_norm[:,:,None,:,None].expand(-1,-1,L,-1,-1)
        cf=m3_conf[:,:,:,None,None].expand(-1,-1,-1,self.P,1)
        local=torch.stack([rr_norm,torch.abs(rr_norm)],-1)
        o=self.role(torch.cat([pe,he,local,be,cf],-1))
        support_logits=o[...,0]; scale=torch.sigmoid(o[...,1])*self.max_scale
        qh=q[:,:,None,:].expand(-1,-1,self.P,-1); qsh=qs[:,:,None,:].expand(-1,-1,self.P,-1)
        hh=self.h_emb.weight[None,None,:,:].expand(B,N,-1,-1)
        noop=self.noop(torch.cat([qh,qsh,hh,base_norm[:,:,:,None]],-1)).squeeze(-1)
        return support_logits,scale,noop


class PairwiseSupportRouter(nn.Module):
    """v0.7 query-support compatibility router.

    The v0.6 diagnosis showed that the SoftMix teacher is strong while the compact
    student representation is not learnable enough. v0.7 therefore exposes the
    complete deployable node context of both query and historical support, the full
    historical residual trajectory, candidate-relative M3 confidence, and the
    query/support PatchSTG forecast relation before node-horizon routing.

    All inputs remain deployment-safe. Historical support residuals and support
    contexts come only from Train memory; current-query future targets are never inputs.
    """
    def __init__(self,demb,node_feature_dim,P,sense_dim=6,context_dim=96,residual_dim=48,
                 pair_dim=128,hidden=192,horizon_dim=16,attention_heads=4,
                 max_scale=1.5,dropout=0.1):
        super().__init__(); self.P=int(P); self.demb=int(demb); self.node_feature_dim=int(node_feature_dim)
        self.context_dim=int(context_dim); self.residual_dim=int(residual_dim); self.pair_dim=int(pair_dim)
        self.max_scale=float(max_scale); self.hdim=int(horizon_dim)
        self.qctx=nn.Sequential(nn.LayerNorm(self.node_feature_dim),nn.Linear(self.node_feature_dim,self.context_dim),nn.GELU(),nn.Dropout(float(dropout)),nn.Linear(self.context_dim,self.context_dim),nn.GELU())
        self.sctx=nn.Sequential(nn.LayerNorm(self.node_feature_dim),nn.Linear(self.node_feature_dim,self.context_dim),nn.GELU(),nn.Dropout(float(dropout)),nn.Linear(self.context_dim,self.context_dim),nn.GELU())
        self.residual=nn.Sequential(nn.LayerNorm(2*self.P),nn.Linear(2*self.P,self.residual_dim),nn.GELU(),nn.Dropout(float(dropout)),nn.Linear(self.residual_dim,self.residual_dim),nn.GELU())
        pair_in=4*self.demb + 4*self.context_dim + 3*int(sense_dim) + self.residual_dim + 1
        self.pair=nn.Sequential(nn.Linear(pair_in,int(hidden)),nn.LayerNorm(int(hidden)),nn.GELU(),nn.Dropout(float(dropout)),nn.Linear(int(hidden),self.pair_dim),nn.GELU())
        heads=max(1,int(attention_heads))
        if self.pair_dim % heads != 0: heads=1
        self.cross=nn.MultiheadAttention(self.pair_dim,heads,dropout=float(dropout),batch_first=True)
        self.cross_norm=nn.LayerNorm(self.pair_dim)
        self.h_emb=nn.Embedding(self.P,self.hdim)
        # local features: residual, |residual|, query base, support base,
        # query-support base difference, M3 relative confidence.
        role_in=self.pair_dim+self.hdim+6
        self.role=nn.Sequential(nn.Linear(role_in,int(hidden)),nn.GELU(),nn.Dropout(float(dropout)),nn.Linear(int(hidden),96),nn.GELU(),nn.Linear(96,2))
        noop_in=self.demb+self.context_dim+int(sense_dim)+self.hdim+1
        self.noop=nn.Sequential(nn.Linear(noop_in,96),nn.GELU(),nn.Dropout(float(dropout)),nn.Linear(96,1))

    def forward(self,q,c,qctx,cctx,qs,cs,rr_norm,base_norm,support_base_norm,m3_conf):
        # q [B,N,D], c [B,N,L,D]
        # qctx [B,N,F], cctx [B,N,L,F]
        # rr_norm/support_base_norm [B,N,L,P], base_norm [B,N,P]
        B,N,L,_=c.shape
        qcx=self.qctx(qctx); scx=self.sctx(cctx)
        qe=q[:,:,None,:].expand(-1,-1,L,-1); qce=qcx[:,:,None,:].expand(-1,-1,L,-1); qse=qs[:,:,None,:].expand(-1,-1,L,-1)
        rseq=self.residual(torch.cat([rr_norm,torch.abs(rr_norm)],-1))
        pair_in=torch.cat([
            qe,c,torch.abs(qe-c),qe*c,
            qce,scx,torch.abs(qce-scx),qce*scx,
            qse,cs,torch.abs(qse-cs),
            rseq,m3_conf[:,:,:,None]
        ],-1)
        pair=self.pair(pair_in)
        flat=pair.reshape(B*N,L,self.pair_dim)
        att,_=self.cross(flat,flat,flat,need_weights=False)
        pair=self.cross_norm(flat+att).reshape(B,N,L,self.pair_dim)

        pe=pair[:,:,:,None,:].expand(-1,-1,-1,self.P,-1)
        he=self.h_emb.weight[None,None,None,:,:].expand(B,N,L,-1,-1)
        qb=base_norm[:,:,None,:].expand(-1,-1,L,-1)
        local=torch.stack([rr_norm,torch.abs(rr_norm),qb,support_base_norm,support_base_norm-qb,m3_conf[:,:,:,None].expand(-1,-1,-1,self.P)],-1)
        o=self.role(torch.cat([pe,he,local],-1))
        support_logits=o[...,0]
        scale=torch.sigmoid(o[...,1])*self.max_scale

        qh=q[:,:,None,:].expand(-1,-1,self.P,-1); qch=qcx[:,:,None,:].expand(-1,-1,self.P,-1)
        qsh=qs[:,:,None,:].expand(-1,-1,self.P,-1); hh=self.h_emb.weight[None,None,:,:].expand(B,N,-1,-1)
        noop=self.noop(torch.cat([qh,qch,qsh,hh,base_norm[:,:,:,None]],-1)).squeeze(-1)
        return support_logits,scale,noop


class SensingAwareSafeGate(nn.Module):
    def __init__(self,demb,P,N,sense_dim=6,node_dim=16,horizon_dim=8,hidden=96):
        super().__init__(); self.P,self.N=int(P),int(N)
        self.node_emb=nn.Embedding(self.N,int(node_dim)); self.h_emb=nn.Embedding(self.P,int(horizon_dim))
        din=int(demb)+int(sense_dim)+int(node_dim)+int(horizon_dim)+5
        self.net=nn.Sequential(nn.Linear(din,int(hidden)),nn.GELU(),nn.Dropout(0.1),nn.Linear(int(hidden),48),nn.GELU(),nn.Linear(48,1))
        nn.init.zeros_(self.net[-1].weight); nn.init.constant_(self.net[-1].bias,0.5)
    def forward(self,q,qs,base,corr,support_mass,disagreement):
        B,P,N=base.shape
        if P!=self.P or N!=self.N: raise ValueError(f'Gate shape mismatch {(P,N)} vs {(self.P,self.N)}')
        qe=q[:,None,:,:].expand(B,P,N,-1); qse=qs[:,None,:,:].expand(B,P,N,-1)
        ne=self.node_emb.weight[None,None,:,:].expand(B,P,N,-1); he=self.h_emb.weight[None,:,None,:].expand(B,P,N,-1)
        local=torch.stack([base,corr,torch.abs(corr),support_mass,disagreement],-1)
        return torch.sigmoid(self.net(torch.cat([qe,qse,ne,he,local],-1)).squeeze(-1))
