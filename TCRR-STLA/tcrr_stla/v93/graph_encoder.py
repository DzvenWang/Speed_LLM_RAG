"""CF-only spatial graph encoding. No independent graph forecast correction.

Original v7 road graph is stored as versioned source under v93/road_graph.py.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from .road_graph import load_or_build_graph


def functional_graph(cache,k=8,max_windows=1024):
    """v7 pairwise-valid functional graph, trained ONLY from the train cache."""
    from pathlib import Path
    root=Path(cache)
    a=np.load(root/'train_x_full_norm.npy',mmap_mode='r')
    obs=np.load(root/'train_history_observed.npy',mmap_mode='r')
    artificial=np.load(root/'train_artificial_missing.npy',mmap_mode='r') if (root/'train_artificial_missing.npy').exists() else None
    ids=np.linspace(0,len(a)-1,min(max_windows,len(a)),dtype=np.int64)
    x=np.asarray(a[ids],dtype=np.float32)
    valid=np.asarray(obs[ids],dtype=bool)
    if artificial is not None:valid|=np.asarray(artificial[ids],dtype=bool)
    valid &= np.isfinite(x)
    X=torch.from_numpy(np.nan_to_num(x,nan=0,posinf=0,neginf=0))
    M=torch.from_numpy(valid)
    # Taken from v7's exact pairwise overlap-shrunk correlation formulation.
    from ..v93.functional import make_functional_neighbours
    return make_functional_neighbours(X,M,k=k)


class CFGraphEncoder(nn.Module):
    def __init__(self,mode,nodes,hidden,road=None,functional=None,edge_dim=16):
        super().__init__()
        if mode not in ('none','fixed','v7_edge_attention'):
            raise ValueError(f'Unsupported graph mode {mode}')
        self.mode=mode;self.hidden=hidden;self.n=int(nodes)
        self.relations=('functional','upstream','downstream')
        if mode=='none':return
        if road is None or functional is None:raise ValueError('Graph mode requires v7 road + functional graphs')
        fids,fvalid=functional
        if fids.shape[0]!=nodes or fvalid.shape!=fids.shape:raise ValueError('Functional graph node mismatch')
        data={'functional_ids':fids,'functional_valid':fvalid}
        for side in ('upstream','downstream'):
            for att in ('ids','valid','gps_log','road_log','road_valid'):
                data[side+'_'+att]=road[side+'_'+att]
        for r in self.relations:
            if r=='functional':
                ids=torch.as_tensor(data[r+'_ids'],dtype=torch.long)
                valid=torch.as_tensor(data[r+'_valid'],dtype=torch.bool)
                gps=torch.zeros_like(ids,dtype=torch.float32)
                dist=torch.zeros_like(ids,dtype=torch.float32)
                dvalid=torch.zeros_like(ids,dtype=torch.bool)
            else:
                ids=torch.as_tensor(data[r+'_ids'],dtype=torch.long)
                valid=torch.as_tensor(data[r+'_valid'],dtype=torch.bool)
                gps=torch.as_tensor(data[r+'_gps_log'],dtype=torch.float32)
                dist=torch.as_tensor(data[r+'_road_log'],dtype=torch.float32)
                dvalid=torch.as_tensor(data[r+'_road_valid'],dtype=torch.bool)
            if ids.ndim!=2 or ids.shape[0]!=nodes or ids.shape!=valid.shape:
                raise ValueError(f'{r} ids invalid')
            if valid.any() and (ids[valid].min()<0 or ids[valid].max()>=nodes):
                raise ValueError(f'{r} neighbour indices invalid')
            self.register_buffer(r+'_ids',ids)
            self.register_buffer(r+'_valid',valid)
            self.register_buffer(r+'_edge',torch.stack([gps,dist,dvalid.float(),valid.float()],-1))
        if mode=='v7_edge_attention':
            self.query=nn.Linear(hidden,edge_dim,bias=False)
            self.key=nn.Linear(hidden,edge_dim,bias=False)
            self.edge=nn.Linear(4,edge_dim)
            self.horizon=nn.Embedding(12,edge_dim)
            self.value=nn.Linear(hidden,hidden)
        self.fusion=nn.Sequential(nn.Linear(hidden*4,hidden),nn.GELU(),nn.LayerNorm(hidden))

    def _message(self,z,r,coverage):
        # z: [B,H,N,D], coverage: [B,N] from visible input.
        ids=getattr(self,r+'_ids')
        valid=getattr(self,r+'_valid')
        edge=getattr(self,r+'_edge')
        B,H,N,D=z.shape
        if ids.shape[1]==0 or not bool(valid.any()):return z.new_zeros(z.shape)
        src=z[:,:,ids,:]  # [B,H,N,K,D]
        if self.mode=='fixed':
            w=valid.float()[None,None].expand(B,H,-1,-1)
            # Prefer observed neighbours, but do not forbid unobserved ones.
            w=w*(0.25+0.75*coverage[:,None,ids])
        else:
            q=self.query(z)[:,:,:,None,:]
            k=self.key(src)
            logits=(q*k).sum(-1)/(self.query.out_features**.5)
            edge_scores=torch.einsum('hd,nkd->hnk',
                           self.horizon.weight[:H],self.edge(edge))
            logits=logits+edge_scores[None]+torch.log((.25+.75*coverage[:,None,ids]).clamp_min(1e-6))
            logits=logits.float().masked_fill(~valid[None,None],-1e4)
            w=torch.softmax(logits,-1)*valid[None,None].float()
        w=w/w.sum(-1,keepdim=True).clamp_min(1e-8)
        if self.mode=='fixed':src_out=src
        else:src_out=self.value(src)
        return (src_out*w[...,None]).sum(3)

    def forward(self,z,observed):
        if self.mode=='none':return torch.zeros_like(z)
        if z.shape[2]!=self.n:raise RuntimeError('Graph node dimension mismatch')
        coverage=observed.float().mean(1)
        f=[self._message(z,r,coverage) for r in self.relations]
        return self.fusion(torch.cat((z,*f),dim=-1))
