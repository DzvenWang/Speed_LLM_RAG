"""Fixed train-only functional and optional physically grounded directed graph.

Graph orientation: matrix A[i,j] means node j is evidence for target node i.
Never infer road direction from array index.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ..traffic_neighbors import train_only_neighbours


def row_normalize(a:torch.Tensor)->torch.Tensor:
    a=torch.where(torch.isfinite(a),a,torch.zeros_like(a)).clamp_min(0.)
    return a/a.sum(-1,keepdim=True).clamp_min(1e-9)


def index_to_adj(index:torch.Tensor,n:int)->torch.Tensor:
    if index.ndim!=2 or index.shape[0]!=n:raise ValueError('neighbour size mismatch')
    a=torch.zeros(n,n,dtype=torch.float32)
    valid=(index>=0)&(index<n)
    i=torch.arange(n)[:,None].expand_as(index)
    a[i[valid],index[valid].long()]=1
    a.fill_diagonal_(0)
    return row_normalize(a)

class GraphBundle(nn.Module):
    def __init__(self,functional,upstream=None,downstream=None,functional_sign=None):
        super().__init__();n=functional.shape[0]
        if functional.shape!=(n,n):raise ValueError('Expected square functional graph')
        for key,a in [('functional',functional),('upstream',upstream),('downstream',downstream)]:
            a=torch.zeros_like(functional) if a is None else a
            if a.shape!=(n,n):raise ValueError(f'{key} mismatch')
            self.register_buffer(key,row_normalize(a.float()))
        if functional_sign is None:functional_sign=torch.ones_like(functional)
        if functional_sign.shape!=(n,n):raise ValueError('Functional sign shape mismatch')
        self.register_buffer('functional_signed',self.functional*torch.sign(functional_sign.float()))
        # Sparse buffers avoid dense O(N^2) historical-CF graph propagation.
        self.register_buffer('functional_sparse',self.functional_signed.to_sparse().coalesce())
        self.register_buffer('upstream_sparse',self.upstream.to_sparse().coalesce())
        self.register_buffer('downstream_sparse',self.downstream.to_sparse().coalesce())
        self.nodes=n
        self.physical_available=(upstream is not None and downstream is not None)

    @classmethod
    def from_cache(cls,cache,n,physical_npz=None,k=8):
        neigh=train_only_neighbours(cache,k)
        f=index_to_adj(neigh,n)
        # Preserve sign for negative-correlated functional neighbors in CF spatial evidence.
        hist=np.load(Path(cache)/'train_x_full_norm.npy',mmap_mode='r')
        rows=np.linspace(0,len(hist)-1,min(len(hist),1200),dtype=np.int64)
        raw=np.nan_to_num(np.array(hist[rows],np.float32,copy=True))
        z=np.diff(raw,axis=1).reshape(-1,n).astype(np.float32)
        z-=z.mean(0,keepdims=True)
        z/=np.sqrt((z*z).mean(0,keepdims=True)+1e-6)
        sign=torch.ones(n,n,dtype=torch.float32)
        selected=neigh.cpu().numpy()
        for i in range(n):
            corr=(z[:,i,None]*z[:,selected[i]]).mean(0)
            sign[i,selected[i]]=torch.from_numpy(np.sign(corr).astype('float32'))
        up=down=None
        if physical_npz:
            with np.load(physical_npz,allow_pickle=False) as z:
                keys=set(z.files)
                if {'upstream','downstream'}<=keys:
                    up=torch.from_numpy(np.asarray(z['upstream'],np.float32))
                    down=torch.from_numpy(np.asarray(z['downstream'],np.float32))
                elif {'upstream_edges','downstream_edges'}<=keys:
                    def edges_to_matrix(edges):
                        a=torch.zeros(n,n)
                        for i,j in edges:
                            if not (0<=i<n and 0<=j<n):raise ValueError('graph edge out of bounds')
                            a[int(i),int(j)]=1
                        return a
                    up=edges_to_matrix(z['upstream_edges']);down=edges_to_matrix(z['downstream_edges'])
                else:raise ValueError('physical graph NPZ requires upstream/downstream matrices or *_edges')
                if 'node_ids' not in z:
                    raise ValueError('physical graph requires node_ids for ordering verification')
                if len(z['node_ids'])!=n:raise ValueError('graph node count mismatch')
                # Ordered node IDs must additionally be compared with dataset IDs by runner.
        return cls(f,up,down,functional_sign=sign)

    def matrices(self):return (self.functional,self.upstream,self.downstream)

    def query_summary(self,x,mask):
        """x/mask [B,L,N], no hidden truth. output [B,N,9]"""
        if x.shape!=mask.shape or x.shape[-1]!=self.nodes:raise ValueError('query shape mismatch')
        recent=mask[:,-min(3,mask.shape[1]):].float().mean(1)
        mean_visible=(x*mask).sum(1)/mask.sum(1).clamp_min(1.)
        outputs=[]
        for a in self.matrices():
            exists=(a.sum(-1)>0).float()[None,:].expand(x.shape[0],-1)
            mass=torch.einsum('ij,bj->bi',a,recent)
            weighted=torch.einsum('ij,bj->bi',a,mean_visible*recent)
            state=weighted/mass.clamp_min(1e-6)
            outputs.extend([state,mass,exists])
        return torch.stack(outputs,-1)

    def cf_summary(self,cf):
        """cf [B,K,H,N] -> [B,K,H,N,3] graph responses, no train labels."""
        if cf.shape[-1]!=self.nodes:raise ValueError('CF nodes mismatch')
        # CUDA sparse GEMM dtype/casting behavior depends on torch release.
        # Keep this deterministic and FP32 even under v9 autocast FP16.
        with torch.autocast(device_type=cf.device.type,enabled=False):
            flat=cf.float().reshape(-1,self.nodes).T.contiguous()  # [N,B*K*H]
            outputs=[]
            for a in (self.functional_sparse,self.upstream_sparse,self.downstream_sparse):
                out=torch.sparse.mm(a.float(),flat).T.reshape(*cf.shape)
                outputs.append(out)
            return torch.stack(outputs,-1)
