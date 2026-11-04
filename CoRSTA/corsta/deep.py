"""CoRSTA: differentiable C1 feature evidence, horizon-dynamic C2, STLA feedback.

Only query history, masks and fixed TRAIN-derived candidate features enter forward.
Training targets are never arguments to the prediction model.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .student import SplitSTLA
from .v7.model import SupportAwareTCRR


class C1SupportTokens(nn.Module):
    def __init__(self, model_dim:int, token_dim:int=32):
        super().__init__()
        self.horizon_emb=nn.Embedding(512,8)
        self.history=nn.Sequential(nn.Linear(model_dim+4,32),nn.GELU(),nn.Linear(32,16),nn.GELU())
        self.normal=nn.Sequential(nn.Linear(5+16+8,token_dim),nn.GELU(),nn.Linear(token_dim,token_dim))
        self.cf=nn.Sequential(nn.Linear(5+16+8,token_dim),nn.GELU(),nn.Linear(token_dim,token_dim))
        self.cf_gate=nn.Sequential(nn.Linear(5,16),nn.GELU(),nn.Linear(16,1))
        self.confidence=nn.Sequential(nn.Linear(5,16),nn.GELU(),nn.Linear(16,1))
        self.norm=nn.LayerNorm(token_dim)

    def forward(self,out,p0,h0,mask):
        B,P,N=p0.shape;K=out['candidate_weights'].shape[1]-1
        weight=out['candidate_weights'][:,:K]
        valid=out['candidate_available'].float()
        normal=(out['normal_component']*weight*valid).sum(1)
        cf=(out['cf_increment']*weight*valid).sum(1)
        none=out['none_weight']
        comp=out['support_compatibility']
        variance=torch.log1p(out['support_variance'].clamp_min(0))
        observed=mask.float().mean(1)[:,None].expand(-1,P,-1)
        context=self.history(torch.cat((h0.mean(1),mask.mean(1)[...,None],
            mask[:,-1,:,None],mask[:,:,:,None].mean(1),mask[:,:,:,None].float().std(1,unbiased=False)),-1))
        context=context[:,None].expand(-1,P,-1,-1)
        hz=self.horizon_emb(torch.arange(P,device=p0.device))[None,:,None].expand(B,-1,N,-1)
        reliability=torch.stack((1-none,comp,torch.exp(-variance),observed,
                                 (out['cf_response'].abs()+1e-5).tanh()),-1)
        main=torch.stack((normal,out['normal_response'],p0,observed,comp),-1)
        anti=torch.stack((cf,out['cf_response'],p0,1-none,variance),-1)
        token_n=self.normal(torch.cat((main,context,hz),-1))
        token_cf=self.cf(torch.cat((anti,context,hz),-1))
        g=torch.sigmoid(self.cf_gate(reliability))
        confidence=torch.sigmoid(self.confidence(reliability)).squeeze(-1)*(1-none)
        token=self.norm(token_n+g*token_cf)
        return token*confidence[...,None],confidence,{'cf_gate':g.squeeze(-1),'none':none}


class HorizonGraphMessage(nn.Module):
    def __init__(self,dim:int,att_dim:int=16):
        super().__init__()
        self.q=nn.Linear(dim,att_dim,bias=False)
        self.k=nn.Linear(dim,att_dim,bias=False)
        self.v=nn.Linear(dim,dim,bias=False)
        self.edge=nn.Linear(4,1)
        self.reliability=nn.Sequential(nn.Linear(2,8),nn.GELU(),nn.Linear(8,1))
        self.dim=dim;self.att_dim=att_dim

    def forward(self,z,ids,valid,obs,support,gps=None,road=None,road_valid=None):
        B,P,N,D=z.shape;K=ids.shape[1]
        if K==0 or not valid.any():return z.new_zeros(z.shape)
        src=z[:,:,ids,:]   # [B,P,N,K,D] — per-horizon, unlike old C2
        raw=(self.q(z)[:,:,:,None,:]*self.k(src)).sum(-1)/math.sqrt(self.att_dim)
        def ed(v):
            return torch.zeros((N,K),device=z.device,dtype=z.dtype) if v is None else v.to(z.dtype)
        edge=torch.stack((ed(gps),ed(road),ed(road_valid),valid.to(z.dtype)),-1)
        bias=self.edge(edge).squeeze(-1)
        r=torch.stack((obs[:,:,ids],support[:,:,ids]),-1)
        bias_r=self.reliability(r).squeeze(-1)
        logits=(raw+bias[None,None]+bias_r).float().masked_fill(~valid[None,None],-1e4)
        weights=torch.softmax(logits,dim=-1)*valid[None,None]
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
        msg=(weights[...,None]*self.v(src)).sum(-2)
        return msg


class C2ReliabilityGraph(nn.Module):
    def __init__(self, graph, model_dim:int, support_dim:int=32, graph_dim:int=32):
        super().__init__()
        self.horizon_embedding=nn.Embedding(512,8)
        self.current=nn.Sequential(nn.Linear(model_dim+support_dim+4+8,graph_dim),
                                   nn.GELU(),nn.Linear(graph_dim,graph_dim))
        self.message=HorizonGraphMessage(graph_dim)
        self.graph_gate=nn.Sequential(nn.Linear(graph_dim*4+2,graph_dim),nn.GELU(),nn.Linear(graph_dim,3))
        self.out=nn.Sequential(nn.Linear(graph_dim*4,graph_dim),nn.GELU(),nn.Linear(graph_dim,graph_dim))
        self.conf=nn.Sequential(nn.Linear(graph_dim+2,16),nn.GELU(),nn.Linear(16,1))
        self.register_buffer('functional_ids',torch.as_tensor(graph['functional']).long())
        self.register_buffer('functional_valid',torch.as_tensor(graph['functional_valid']).bool())
        for direction in ('upstream','downstream'):
            for k in ('ids','valid','gps_log','road_log','road_valid'):
                v=torch.as_tensor(graph['road'][f'{direction}_{k}'])
                v=v.long() if k=='ids' else v.bool() if k in ('valid','road_valid') else v.float()
                self.register_buffer(f'{direction}_{k}',v)

    def forward(self,h0,p0,s1,sup_rel,mask):
        B,P,N=p0.shape
        observed=mask.float().mean(1)[:,None].expand(-1,P,-1)
        last=mask[:,-1].float()[:,None].expand(-1,P,-1)
        hmean=h0.mean(1)[:,None].expand(-1,P,-1,-1)
        horizon=self.horizon_embedding(torch.arange(P,device=p0.device))[None,:,None].expand(B,-1,N,-1)
        z=self.current(torch.cat((hmean,s1,p0[...,None],observed[...,None],
                                  last[...,None],sup_rel[...,None],horizon),-1))
        func=self.message(z,self.functional_ids,self.functional_valid,observed,sup_rel)
        up=self.message(z,self.upstream_ids,self.upstream_valid,observed,sup_rel,
                        self.upstream_gps_log,self.upstream_road_log,self.upstream_road_valid)
        down=self.message(z,self.downstream_ids,self.downstream_valid,observed,sup_rel,
                        self.downstream_gps_log,self.downstream_road_log,self.downstream_road_valid)
        allmsg=torch.cat((z,func,up,down),-1)
        mix=torch.softmax(self.graph_gate(torch.cat((allmsg,observed[...,None],sup_rel[...,None]),-1)),dim=-1)
        mixed=mix[...,0,None]*func+mix[...,1,None]*up+mix[...,2,None]*down
        token=self.out(torch.cat((z,mixed,up,down),-1))
        confidence=torch.sigmoid(self.conf(torch.cat((token,observed[...,None],sup_rel[...,None]),-1))).squeeze(-1)
        return token*confidence[...,None],confidence,{'graph_mix':mix}


class TemporalBridge(nn.Module):
    def __init__(self,model_dim,source_dim,heads=4):
        super().__init__()
        if model_dim%heads:raise ValueError('model_dim must be divisible by heads')
        self.source_proj=nn.Linear(source_dim,model_dim)
        self.history_pos=nn.Embedding(512,model_dim)
        self.future_pos=nn.Embedding(512,model_dim)
        self.attention=nn.MultiheadAttention(model_dim,heads,batch_first=True)
        self.out=nn.Linear(model_dim,model_dim)

    def forward(self,h,source):
        B,L,N,D=h.shape;P=source.shape[1]
        if source.shape[:3]!=(B,P,N):raise ValueError('Bridge node/batch mismatch')
        query=h.permute(0,2,1,3).reshape(B*N,L,D)
        keys=self.source_proj(source).permute(0,2,1,3).reshape(B*N,P,D)
        query=query+self.history_pos(torch.arange(L,device=h.device))[None]
        keys=keys+self.future_pos(torch.arange(P,device=h.device))[None]
        y,_=self.attention(query,keys,keys,need_weights=False)
        return self.out(y).reshape(B,N,L,D).permute(0,2,1,3)


class ReliabilityFusion(nn.Module):
    def __init__(self,model_dim:int):
        super().__init__()
        self.g1=nn.Sequential(nn.Linear(model_dim+3,32),nn.GELU(),nn.Linear(32,1))
        self.g2=nn.Sequential(nn.Linear(model_dim+3,32),nn.GELU(),nn.Linear(32,1))
        nn.init.constant_(self.g1[-1].bias,-3.)
        nn.init.constant_(self.g2[-1].bias,-3.)
        self.scale=nn.Parameter(torch.tensor(0.1))

    def forward(self,h,a1,a2,r1,r2,mask):
        B,L,N,D=h.shape
        r1_hist=r1.mean(1)[:,None,:,None].expand(-1,L,-1,-1)
        r2_hist=r2.mean(1)[:,None,:,None].expand(-1,L,-1,-1)
        cover=mask.float().mean(1)[:,None,:,None].expand(-1,L,-1,-1)
        common=torch.cat((h,r1_hist,r2_hist,cover),-1)
        gate1=torch.sigmoid(self.g1(common))*r1_hist
        gate2=torch.sigmoid(self.g2(common))*r2_hist
        adapted=h+self.scale*(gate1*a1+gate2*a2)
        return adapted,{'gate_c1':gate1.mean(),'gate_c2':gate2.mean()}


class CoRSTA(nn.Module):
    def __init__(self,student:SplitSTLA,legacy:SupportAwareTCRR,graph,
                 token_dim=32,graph_dim=32,heads=4):
        super().__init__()
        self.student=student
        self.c1=legacy
        D=student.model_dim
        self.support=C1SupportTokens(D,token_dim)
        self.c2=C2ReliabilityGraph(graph,D,token_dim,graph_dim)
        self.bridge1=TemporalBridge(D,token_dim,heads)
        self.bridge2=TemporalBridge(D,graph_dim,heads)
        self.fusion=ReliabilityFusion(D)

    def configure_student(self,phase='partial',last_layers=2):
        self.student.set_train_phase(phase,last_layers)

    def forward(self,b):
        x=b['x'];mask=b['mask'];tod=b['tod'];dow=b['dow']
        prefix,h0,p0,context=self.student.forward_features(x,mask,tod,dow)
        c1=self.c1.forward_progressive(p0,h0.mean(1),x,mask,
             b['candidate_normal'],b['candidate_cf'],
             candidate_scores=b['candidate_scores'],candidate_coverage=b['candidate_coverage'],
             candidate_valid=b['candidate_valid'],candidate_support_raw=b['candidate_support_raw'],
             max_stage=1)
        token1,rel1,aux1=self.support(c1,p0,h0,mask)
        token2,rel2,aux2=self.c2(h0,p0,token1,rel1,mask)
        bridge1=self.bridge1(prefix,token1)
        bridge2=self.bridge2(prefix,token2)
        adapted,gates=self.fusion(prefix,bridge1,bridge2,rel1,rel2,mask)
        refined=self.student.forecast_head(self.student.encode_tail(adapted,context))
        return {'backbone':p0,'refined':refined,'c1_posthoc':c1['c1'],
                'legacy':c1,'reliability_c1':rel1,'reliability_c2':rel2,
                'aux_c1':aux1,'aux_c2':aux2,'gates':gates}
