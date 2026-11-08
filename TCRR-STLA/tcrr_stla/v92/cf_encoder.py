"""Local temporal and relation-specific counterfactual encoders.

The feature tensors are block-local [B,K,G,N,D], not enormous [B,K,H,N,D].
Graph orientation is A[target,source]. Physical edges are optional, never inferred
from array indices. Query-mask reliability is a *learned feature*, not a hard
multiplicative attenuation of historical responses.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class TemporalCFEncoder(nn.Module):
    def __init__(self, blocks=3, hidden=16):
        super().__init__()
        self.blocks=blocks
        self.conv=nn.Sequential(nn.Conv1d(2,12,3,padding=1),nn.GELU(),nn.Conv1d(12,1,3,padding=1))
        self.mlp=nn.Sequential(nn.Linear(11,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())

    def forward(self,normal,cf,mode='cnn'):
        if normal.shape!=cf.shape or cf.ndim!=4:raise ValueError('Normal/CF must match [B,K,H,N]')
        b,k,h,n=cf.shape
        if h%self.blocks:raise ValueError('Horizon must divide blocks')
        span=h//self.blocks
        d=cf.reshape(b,k,self.blocks,span,n)
        r=normal.reshape(b,k,self.blocks,span,n)
        if mode=='cnn':
            inp=torch.stack((normal,cf),-1).permute(0,1,3,4,2).reshape(b*k*n,2,h)
            conv=self.conv(inp).reshape(b,k,n,h).permute(0,1,3,2).reshape(b,k,self.blocks,span,n)
            cm=conv.mean(3);ct=conv[:,:,:,-1]-conv[:,:,:,0]
        elif mode=='stats':
            cm=torch.zeros_like(d.mean(3));ct=torch.zeros_like(cm)
        else:raise ValueError(f'Unknown temporal mode {mode}')
        stats=torch.stack((d.mean(3),d.abs().mean(3),d.std(3,unbiased=False),
                d[:,:,:,-1]-d[:,:,:,0],r.mean(3),r.abs().mean(3),
                (d*r).mean(3),cm,ct,d.amax(3)-d.amin(3),
                ((d[:, :, :, 1:]-d[:, :, :, :-1]).abs().mean(3) if span>1 else torch.zeros_like(cm))),-1)
        return self.mlp(stats)

class MultiRelationCFGraph(nn.Module):
    """One sparse message pass per graph relation, with learned query-conditioned gates."""
    def __init__(self,hidden=16,query_dim=32):
        super().__init__();self.hidden=hidden
        self.message=nn.ModuleList(nn.Linear(hidden,hidden,bias=False) for _ in range(3))
        self.gating=nn.Sequential(nn.Linear(query_dim+6,32),nn.GELU(),nn.Linear(32,3))
        self.fuse=nn.Sequential(nn.Linear(hidden,hidden,bias=False),nn.GELU(),nn.Linear(hidden,hidden,bias=False),nn.GELU())

    def forward(self,local,query,graph_summary,graph,mode='gnn_reliability'):
        if local.ndim!=5 or query.ndim!=3 or graph_summary.shape[-1]!=9:
            raise ValueError('Invalid graph module shapes')
        b,k,g,n,d=local.shape
        if d!=self.hidden or query.shape[:2]!=(b,n) or graph.nodes!=n:
            raise ValueError('Graph input node/hidden mismatch')
        zero=local.new_zeros((b,k,g,n,d))
        gate_zero=query.new_zeros((b,n,3))
        if mode=='none':return zero,gate_zero
        if mode not in ('fixed','gnn','gnn_reliability'):raise ValueError(f'Invalid graph mode {mode}')
        masses=graph_summary[..., [1,4,7]].float()
        present=graph_summary[..., [2,5,8]].float()
        available=present
        if mode=='gnn_reliability':
            gates=torch.sigmoid(self.gating(torch.cat([query.float(),masses,present],-1))) * available
        else:
            gates=available
        parts=[]
        matrices=(graph.functional_sparse,graph.upstream_sparse,graph.downstream_sparse)
        with torch.autocast(device_type=local.device.type,enabled=False):
            for i,mat in enumerate(matrices):
                if mode=='fixed':hmsg=local.float()
                else:hmsg=self.message[i](local.float())
                flat=hmsg.reshape(b*k*g,n,d).permute(1,0,2).reshape(n,-1).contiguous()
                propagated=torch.sparse.mm(mat.float(),flat)
                response=propagated.reshape(n,b*k*g,d).permute(1,0,2).reshape(b,k,g,n,d)
                parts.append(response)
            weighted=sum(part*gates[:,None,None,:,i:i+1].float() for i,part in enumerate(parts))
            total=gates.float().sum(-1).clamp_min(1.)[:,None,None,:,None]
            combined=(weighted/total).to(local.dtype)
        if mode=='fixed':return combined,gates
        # Graph branch must be caused by ACTUAL neighbour messages, not a
        # second local MLP that remains active with zero/absent neighbours.
        active=(gates.sum(-1)>0).to(local.dtype)[:,None,None,:,None]
        return self.fuse(combined)*active,gates
