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


class PredictiveUtilityRerankerV09A(nn.Module):
    """Unified node-horizon predictive-utility reranker for PreSuRe v0.9A.

    The model scores the existing leakage-safe 24-candidate pool directly, without
    the legacy M3 Top-4 compression.  It uses only deployment-safe query/support
    state, historical Train residuals, and coarse retrieval confidence.  Current
    query future targets are used only by the training loss outside this module.

    Outputs for every [query,node,candidate,horizon] cell:
      * mu: expected relative predictive utility;
      * sigma: aleatoric/selection uncertainty (>0);
      * beneficial_logit: calibrated auxiliary probability logit.
    """
    def __init__(
        self,
        node_feature_dim,
        num_nodes,
        P,
        encoder_hidden=160,
        state_dim=64,
        node_dim=24,
        sense_dim=6,
        residual_dim=48,
        pair_dim=128,
        hidden=192,
        horizon_dim=16,
        attention_heads=4,
        dropout=0.1,
        min_sigma=0.03,
        max_sigma=2.0,
    ):
        super().__init__()
        self.P = int(P)
        self.N = int(num_nodes)
        self.F = int(node_feature_dim)
        self.state_dim = int(state_dim)
        self.sense_dim = int(sense_dim)
        self.pair_dim = int(pair_dim)
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)
        self.encoder = NodeSensingEncoder(
            self.F, self.N, hidden=int(encoder_hidden), demb=self.state_dim,
            node_dim=int(node_dim), dropout=float(dropout),
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(2 * self.P),
            nn.Linear(2 * self.P, int(residual_dim)), nn.GELU(), nn.Dropout(float(dropout)),
            nn.Linear(int(residual_dim), int(residual_dim)), nn.GELU(),
        )
        pair_in = 4 * self.state_dim + 3 * self.sense_dim + int(residual_dim) + 1
        self.pair = nn.Sequential(
            nn.Linear(pair_in, int(hidden)), nn.LayerNorm(int(hidden)), nn.GELU(), nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), self.pair_dim), nn.GELU(),
        )
        heads = max(1, int(attention_heads))
        if self.pair_dim % heads != 0:
            heads = 1
        self.cross = nn.MultiheadAttention(self.pair_dim, heads, dropout=float(dropout), batch_first=True)
        self.cross_norm = nn.LayerNorm(self.pair_dim)
        self.h_emb = nn.Embedding(self.P, int(horizon_dim))
        # local: residual, |residual|, query base, support base,
        # support-query base difference, normalized coarse retrieval score.
        role_in = self.pair_dim + int(horizon_dim) + 6
        self.role = nn.Sequential(
            nn.Linear(role_in, int(hidden)), nn.LayerNorm(int(hidden)), nn.GELU(), nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), 96), nn.GELU(),
        )
        self.mu_head = nn.Linear(96, 1)
        self.sigma_head = nn.Linear(96, 1)
        self.benefit_head = nn.Linear(96, 1)

    def encode_query(self, qctx):
        return self.encoder(qctx)

    def encode_support(self, sctx):
        # sctx [B,N,K,F] -> shared encoder over B*K episodes, retaining node identity.
        B, N, K, F = sctx.shape
        if N != self.N or F != self.F:
            raise ValueError(f'Support context shape {tuple(sctx.shape)} incompatible with N={self.N}, F={self.F}')
        x = sctx.permute(0, 2, 1, 3).reshape(B * K, N, F)
        z = self.encoder(x)
        return z.reshape(B, K, N, self.state_dim).permute(0, 2, 1, 3).contiguous()

    def forward_features(self, qctx, sctx, qs, ss, rr_norm, qbase_norm, sbase_norm, coarse_score):
        """Return the frozen deployment-safe role representation and v0.9A outputs.

        v0.9A.2 uses ``role_features`` only as input to an independent strong-harm
        veto head.  The method introduces no new parameters and therefore keeps
        all existing v0.9A checkpoints state-dict compatible.
        """
        # qctx [B,N,F], sctx [B,N,K,F], rr/sbase [B,N,K,P], qbase [B,P,N]
        B, N, K, _ = sctx.shape
        q = self.encode_query(qctx)
        s = self.encode_support(sctx)
        qe = q[:, :, None, :].expand(-1, -1, K, -1)
        qse = qs[:, :, None, :].expand(-1, -1, K, -1)
        rseq = self.residual(torch.cat([rr_norm, torch.abs(rr_norm)], dim=-1))
        cs = coarse_score[:, None, :, None].expand(-1, N, -1, 1)
        pair = self.pair(torch.cat([
            qe, s, torch.abs(qe - s), qe * s,
            qse, ss, torch.abs(qse - ss),
            rseq, cs,
        ], dim=-1))
        flat = pair.reshape(B * N, K, self.pair_dim)
        att, _ = self.cross(flat, flat, flat, need_weights=False)
        pair = self.cross_norm(flat + att).reshape(B, N, K, self.pair_dim)

        pe = pair[:, :, :, None, :].expand(-1, -1, -1, self.P, -1)
        he = self.h_emb.weight[None, None, None, :, :].expand(B, N, K, -1, -1)
        qb = qbase_norm.permute(0, 2, 1)[:, :, None, :].expand(-1, -1, K, -1)
        csh = coarse_score[:, None, :, None].expand(-1, N, -1, self.P)
        local = torch.stack([
            rr_norm, torch.abs(rr_norm), qb, sbase_norm, sbase_norm - qb, csh,
        ], dim=-1)
        role_features = self.role(torch.cat([pe, he, local], dim=-1))
        mu = self.mu_head(role_features).squeeze(-1)
        raw_sigma = self.sigma_head(role_features).squeeze(-1)
        sigma = torch.nn.functional.softplus(raw_sigma) + self.min_sigma
        sigma = torch.clamp(sigma, max=self.max_sigma)
        beneficial_logit = self.benefit_head(role_features).squeeze(-1)
        return role_features, mu, sigma, beneficial_logit

    def forward(self, qctx, sctx, qs, ss, rr_norm, qbase_norm, sbase_norm, coarse_score):
        _, mu, sigma, beneficial_logit = self.forward_features(
            qctx, sctx, qs, ss, rr_norm, qbase_norm, sbase_norm, coarse_score
        )
        return mu, sigma, beneficial_logit


class StrongHarmVetoHead(nn.Module):
    """Independent v0.9A.2 veto head.

    The v0.9A ranker is frozen.  This head receives its detached 96-D role
    representation plus its frozen numerical outputs and predicts only whether a
    candidate is strongly harmful.  It never re-orders safe candidates; it can
    only veto them.
    """
    def __init__(self, role_dim=96, hidden=96, dropout=0.1):
        super().__init__()
        in_dim = int(role_dim) + 4  # mu, sigma, beneficial logit, frozen rank score
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(hidden)), nn.GELU(), nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(hidden) // 2), nn.GELU(),
            nn.Linear(int(hidden) // 2, 1),
        )

    def forward(self, role_features, mu, sigma, beneficial_logit, rank_score):
        x = torch.cat([
            role_features,
            mu[..., None], sigma[..., None], beneficial_logit[..., None], rank_score[..., None],
        ], dim=-1)
        return self.net(x).squeeze(-1)


class SelectedSupportAcceptanceHead(nn.Module):
    """v0.9A.3 acceptance head for the frozen v0.9A Top-1 support only.

    Input is the frozen Top-1 96-D role representation plus 12 deployment-safe
    ranking-context scalars (utility/risk outputs, rank margins, score dispersion,
    entropy, candidate coverage and relative uncertainty).  This head cannot
    reorder candidates; it predicts only USE-vs-ABSTAIN for the existing Top-1.
    """
    def __init__(self, role_dim=96, context_dim=12, hidden=128, dropout=0.1):
        super().__init__()
        din = int(role_dim) + int(context_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(din),
            nn.Linear(din, int(hidden)), nn.GELU(), nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), int(hidden) // 2), nn.GELU(), nn.Dropout(float(dropout)),
            nn.Linear(int(hidden) // 2, 1),
        )

    def forward(self, selected_features):
        return self.net(selected_features).squeeze(-1)
