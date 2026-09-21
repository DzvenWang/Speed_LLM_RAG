from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,din,hidden,dout,dropout=0.1):
        super().__init__(); self.net=nn.Sequential(nn.Linear(din,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,dout))
    def forward(self,x): return self.net(x)


class DualQueryConditionalForecaster(nn.Module):
    """PreSuRe v1.2.

    Retrieval evidence forms a standalone retrieval-conditioned forecast. Historical
    futures/residuals are encoded as context and are never copied to the query output.
    The final prediction adaptively mixes the protected PatchSTG forecast and the
    retrieval-conditioned forecast at node x horizon resolution.
    """
    def __init__(self,state_dim,evidence_dim,horizon,d_model=96,state_hidden=160,evidence_hidden=192,
                 support_hidden=160,decoder_hidden=192,attention_heads=4,dropout=0.1,
                 selector_prior_weight=0.15,cross_attention_temperature=1.35,
                 alpha_init=0.25,max_delta_sigma=3.0,attention_batch_chunk=32768):
        super().__init__(); self.P=int(horizon); self.d=int(d_model); self.selector_prior_weight=float(selector_prior_weight); self.temperature=float(cross_attention_temperature); self.max_delta_sigma=float(max_delta_sigma); self.attention_batch_chunk=max(1,min(int(attention_batch_chunk),60000))
        self.state_enc=MLP(state_dim,state_hidden,d_model,dropout); self.evidence_enc=MLP(evidence_dim,evidence_hidden,d_model,dropout)
        self.dual_gate=nn.Linear(d_model*2,d_model); self.dual_norm=nn.LayerNorm(d_model)
        self.horizon_emb=nn.Embedding(self.P,d_model); self.base_scalar=MLP(1,48,d_model,dropout)
        self.support_future=MLP(4,64,d_model,dropout); self.score_embed=MLP(1,32,d_model,dropout)
        self.support_self_attn=nn.MultiheadAttention(d_model,int(attention_heads),dropout=dropout,batch_first=True)
        self.support_norm1=nn.LayerNorm(d_model); self.support_ff=nn.Sequential(nn.Linear(d_model,support_hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(support_hidden,d_model)); self.support_norm2=nn.LayerNorm(d_model)
        self.q_proj=nn.Linear(d_model,d_model,bias=False); self.k_proj=nn.Linear(d_model,d_model,bias=False); self.v_proj=nn.Linear(d_model,d_model,bias=False)
        pair_dim=d_model*4
        self.delta_body=nn.Sequential(nn.Linear(pair_dim,decoder_hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(decoder_hidden,decoder_hidden),nn.GELU())
        self.delta_out=nn.Linear(decoder_hidden,1)
        # alpha uses the same conditional pair plus retrieval-confidence statistics.
        self.alpha_body=nn.Sequential(nn.Linear(pair_dim+4,decoder_hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(decoder_hidden,decoder_hidden//2),nn.GELU())
        self.alpha_out=nn.Linear(decoder_hidden//2,1)
        nn.init.zeros_(self.delta_out.weight); nn.init.zeros_(self.delta_out.bias)
        nn.init.zeros_(self.alpha_out.weight); init=float(alpha_init); init=min(max(init,1e-4),1-1e-4); nn.init.constant_(self.alpha_out.bias,math.log(init/(1-init)))

    def _fuse_dual(self,state,evidence):
        s=self.state_enc(state); e=self.evidence_enc(evidence); g=torch.sigmoid(self.dual_gate(torch.cat([s,e],-1))); z=self.dual_norm(g*s+(1.0-g)*e)
        return z,g

    def _support_self_attention_chunked(self,seq,key_padding_mask):
        """Run support-set MHA without exceeding CUDA efficient-attention batch limits.

        PyTorch's efficient scaled-dot-product attention backend cannot emit RNG
        seed/offset metadata when the effective batch dimension exceeds 65,535.
        PreSuRe flattens B x N x P into that dimension; with the formal setting
        B=16, N=388, P=12 this is 74,496.  Chunking only the independent
        flattened batch preserves the exact node-horizon support-attention
        semantics while keeping every backend call below the limit.
        """
        total=int(seq.shape[0]); chunk=self.attention_batch_chunk
        if total<=chunk:
            out,_=self.support_self_attn(seq,seq,seq,key_padding_mask=key_padding_mask,need_weights=False)
            return out
        outs=[]
        for start in range(0,total,chunk):
            end=min(start+chunk,total)
            out,_=self.support_self_attn(seq[start:end],seq[start:end],seq[start:end],key_padding_mask=key_padding_mask[start:end],need_weights=False)
            outs.append(out)
        return torch.cat(outs,dim=0)

    def forward(self,query_state,query_evidence,query_base,support_state,support_evidence,support_future,support_base,support_valid,selector_scores,speed_mean,speed_std):
        std=max(float(speed_std),1e-6); mean=float(speed_mean)
        qb=(query_base-mean)/std; sy=(support_future-mean)/std; sb=(support_base-mean)/std
        sy=torch.nan_to_num(sy,nan=0.0,posinf=0.0,neginf=0.0); sb=torch.nan_to_num(sb,nan=0.0,posinf=0.0,neginf=0.0); sr=sy-sb
        B,P,N=qb.shape; K=sy.shape[1]; h=self.horizon_emb(torch.arange(P,device=qb.device))
        qnode,qgate=self._fuse_dual(query_state,query_evidence) # B,N,D
        qtok=qnode[:,:,None,:]+h[None,None,:,:]+self.base_scalar(qb.permute(0,2,1)[...,None])
        # Shared state/evidence encoders preserve the same semantics for support episodes.
        sstate=self.state_enc(support_state); sev=self.evidence_enc(support_evidence); sg=torch.sigmoid(self.dual_gate(torch.cat([sstate,sev],-1))); snode=self.dual_norm(sg*sstate+(1.0-sg)*sev)
        sf=torch.stack([sy.permute(0,1,3,2),sb.permute(0,1,3,2),sr.permute(0,1,3,2),support_valid.float().permute(0,1,3,2)],dim=-1)
        scores=torch.nan_to_num(selector_scores,nan=0.0,posinf=5.0,neginf=-5.0); scorez=(scores-scores.mean(-1,keepdim=True))/scores.std(-1,keepdim=True).clamp_min(1e-4)
        stok=snode[:,:,:,None,:]+h[None,None,None,:,:]+self.support_future(sf)+self.score_embed(scorez[:,:,None,None,None].expand(-1,-1,N,P,1))
        # Support-set self-attention at each node x horizon. Invalid historical futures are masked.
        seq=stok.permute(0,2,3,1,4).reshape(B*N*P,K,self.d)
        valid=support_valid.permute(0,3,2,1).reshape(B*N*P,K)
        pad=~valid; all_bad=pad.all(1); safe_pad=pad.clone()
        if torch.any(all_bad): safe_pad[all_bad,0]=False
        sa=self._support_self_attention_chunked(seq,safe_pad)
        seq=self.support_norm1(seq+sa); seq=self.support_norm2(seq+self.support_ff(seq)); seq=seq*valid[:,:,None].float()
        if torch.any(all_bad): seq[all_bad]=0.0
        ctx=seq.reshape(B,N,P,K,self.d)
        q=self.q_proj(qtok); k=self.k_proj(ctx); v=self.v_proj(ctx)
        logits=(q[:,:,:,None,:]*k).sum(-1)/math.sqrt(float(self.d)); logits=logits/max(self.temperature,1e-4)
        logits=logits+self.selector_prior_weight*scorez[:,None,None,:]
        vmask=valid.reshape(B,N,P,K); logits=logits.masked_fill(~vmask,-1e4)
        attn=torch.softmax(logits,dim=-1)*vmask.float(); den=attn.sum(-1,keepdim=True); attn=torch.where(den>0,attn/den.clamp_min(1e-8),torch.zeros_like(attn))
        z=(attn[:,:,:,:,None]*v).sum(3)
        pair=torch.cat([qtok,z,qtok-z,qtok*z],-1)
        delta_norm=torch.tanh(self.delta_out(self.delta_body(pair)).squeeze(-1))*self.max_delta_sigma
        ret_pred=query_base.permute(0,2,1)+delta_norm*std # B,N,P
        # Retrieval confidence features: score margin/std, attention entropy, attention max.
        ssort=torch.sort(scorez,dim=-1,descending=True).values; margin=(ssort[:,0]-ssort[:,1] if K>1 else ssort[:,0]); sstd=scorez.std(-1)
        ent=-(attn.clamp_min(1e-8)*attn.clamp_min(1e-8).log()).sum(-1)/math.log(max(K,2)); amax=attn.max(-1).values
        conf=torch.stack([margin[:,None,None].expand(B,N,P),sstd[:,None,None].expand(B,N,P),ent,amax],-1)
        alpha=torch.sigmoid(self.alpha_out(self.alpha_body(torch.cat([pair,conf],-1))).squeeze(-1))
        base_bnp=query_base.permute(0,2,1); fused=(1.0-alpha)*base_bnp+alpha*ret_pred
        return {
            'pred':fused.permute(0,2,1).contiguous(),
            'retrieval_pred':ret_pred.permute(0,2,1).contiguous(),
            'delta':(delta_norm*std).permute(0,2,1).contiguous(),
            'alpha':alpha.permute(0,2,1).contiguous(),
            'attention':attn,
            'attention_entropy':ent,
            'dual_gate_query':qgate,
            'dual_gate_support':sg,
        }
