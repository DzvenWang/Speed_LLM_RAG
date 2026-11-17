"""Synthetic helpers for ReShift unit tests (no official STLA / Gadi fidelity)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

L,P,N=6,4,5


class FakeSTLA(nn.Module):
    """Same interface as SplitSTLA (encode_prefix / encode_tail / forecast_head)."""
    def __init__(self,n=N,h=L,p=P,d=16):
        super().__init__();self.base=nn.Module();self.base.num_nodes=n
        self.base.in_steps=h;self.base.out_steps=p;self.hidden_dim=d
        self.enc=nn.Linear(2,d);self.tail=nn.Linear(d,d);self.output=nn.Linear(h*d,p)
        self.seen_masks=[]
    @property
    def model_dim(self):return self.hidden_dim
    def encode_prefix(self,x,mask,tod,dow):
        self.seen_masks.append(mask.detach().clone())
        return self.enc(torch.stack([x*mask,mask],dim=-1)),()
    def encode_tail(self,h,ctx):return self.tail(h)
    def forecast_head(self,h):
        B,LL,NN,D=h.shape
        return self.output(h.permute(0,2,1,3).reshape(B,NN,LL*D)).permute(0,2,1)
    def forward(self,x,mask,tod,dow):
        h,ctx=self.encode_prefix(x,mask,tod,dow)
        return self.forecast_head(self.encode_tail(h,ctx)),h.mean(1)


def graph(n=N,k=2):
    ids=np.stack([np.roll(np.arange(n),j+1) for j in range(k)],axis=-1).astype(np.int64)
    return {'ids':ids,'valid':np.ones((n,k),bool),'edge_features':np.zeros((n,k,5),np.float32)}


def batch(B=2,seed=0):
    g=torch.Generator().manual_seed(seed)
    x=torch.randn(B,L,N,generator=g)
    m=(torch.rand(B,L,N,generator=g)>.4).float();x=x*m
    obs={'x':x,'mask':m,'tod':torch.zeros(B,L,dtype=torch.long),'dow':torch.zeros(B,L,dtype=torch.long)}
    full=x.clone();full[m==0]=torch.randn(B,L,N,generator=g)[m==0]
    b=dict(obs,x_full=full,original_valid=torch.ones_like(m),
           y_norm=torch.randn(B,P,N,generator=g),y_raw=torch.randn(B,P,N,generator=g),
           future_valid=torch.ones(B,P,N,dtype=torch.bool))
    return obs,b


def write_train_cache(root,count=12,spacing=12,seed=0):
    """Minimal Train cache readable by reshift.retrieval.memory.TrainMemory."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(count,L,N)).astype('float32')
    observed=rng.random((count,L,N))>.3
    artificial=(~observed)&(rng.random((count,L,N))>.5)
    x[~(observed|artificial)]=np.nan          # never observed: must never be read
    arrays={'x_full_norm':x,'history_observed':observed,'artificial_missing':artificial,
            'y_norm':rng.normal(size=(count,P,N)).astype('float32'),
            'future_valid':np.ones((count,P,N),bool),
            'tod_hist':np.zeros((count,L),np.int16),'dow_hist':np.zeros((count,L),np.int16)}
    for k,v in arrays.items():np.save(root/f'train_{k}.npy',v)
    pd.DataFrame({'window_start':np.arange(count)*spacing}).to_csv(root/'train_manifest.csv',index=False)
    return root,arrays


class QueryDataset(Dataset):
    def __init__(self,x,mask,starts,y=None):
        self.x=torch.as_tensor(x).float();self.mask=torch.as_tensor(mask).float()
        self.starts=torch.as_tensor(starts).long()
        self.y=torch.zeros(len(self.x),P,N) if y is None else torch.as_tensor(y).float()
    def __len__(self):return len(self.x)
    def __getitem__(self,i):
        return {'index':torch.tensor(i),'x':self.x[i],'mask':self.mask[i],'window_start':self.starts[i],
                'y_norm':self.y[i]}
