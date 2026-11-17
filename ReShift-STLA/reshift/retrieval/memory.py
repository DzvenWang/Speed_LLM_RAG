"""Train-only labelled memory for failure-replayed retrieval.

Only TRAIN arrays are ever opened here: complete history (for replay), the
original-availability mask, labels and the cached full-evidence forecast of the
frozen STLA.  Validation/Test labels or complete histories are never read.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch

MEMORY_KEYS=('x','ov','tod','dow','y','fv')


class TrainMemory:
    def __init__(self,cache,limit=None):
        root=Path(cache);self.root=root
        mm=lambda name:np.load(root/f'train_{name}.npy',mmap_mode='r')
        self.arrays={'x':mm('x_full_norm'),'observed':mm('history_observed'),
                     'y':mm('y_norm'),'fv':mm('future_valid'),
                     'tod':mm('tod_hist'),'dow':mm('dow_hist')}
        art=root/'train_artificial_missing.npy'
        self.arrays['artificial']=np.load(art,mmap_mode='r') if art.is_file() else None
        meta=pd.read_csv(root/'train_manifest.csv')
        total=min(len(meta),len(self.arrays['x']))
        self.n=total if limit is None else min(int(limit),total)
        if self.n<1:raise RuntimeError('Empty Train memory')
        self.starts=meta.window_start.to_numpy(np.int64)[:self.n]
        self.history=int(self.arrays['x'].shape[1]);self.nodes=int(self.arrays['x'].shape[2])
        self.horizon=int(self.arrays['y'].shape[1])
        self.p_full=None
        self._gpu={}

    def __len__(self):return self.n

    def rows(self,ids):
        """Host rows for integer ids (all must be legal)."""
        ids=np.asarray(ids,dtype=np.int64)
        if len(ids) and (ids.min()<0 or ids.max()>=self.n):raise IndexError('Train memory index out of range')
        a=self.arrays
        ov=np.asarray(a['observed'][ids],dtype=bool)
        if a['artificial'] is not None:ov=ov|np.asarray(a['artificial'][ids],dtype=bool)
        x=np.nan_to_num(np.asarray(a['x'][ids],dtype=np.float32),nan=0.,posinf=0.,neginf=0.)
        x=np.where(ov,x,0.).astype(np.float32)
        y=np.asarray(a['y'][ids],dtype=np.float32)
        fv=np.asarray(a['fv'][ids],dtype=bool)&np.isfinite(y)
        y=np.where(fv,np.nan_to_num(y,nan=0.,posinf=0.,neginf=0.),0.).astype(np.float32)
        return {'x':x,'ov':ov,'tod':np.asarray(a['tod'][ids],dtype=np.int64),
                'dow':np.asarray(a['dow'][ids],dtype=np.int64),'y':y,'fv':fv}

    def attach_p_full(self,array):
        if tuple(array.shape)!=(self.n,self.horizon,self.nodes):
            raise RuntimeError(f'P_full cache shape {array.shape} != {(self.n,self.horizon,self.nodes)}')
        self.p_full=array

    def preload(self,device,keys=MEMORY_KEYS+('p_full',)):
        """Optionally keep memory on the GPU; silently falls back to host memmaps."""
        if torch.device(device).type!='cuda':return False
        try:
            gpu={}
            for start in range(0,self.n,4096):
                rows=self.rows(np.arange(start,min(self.n,start+4096)))
                if self.p_full is not None and 'p_full' in keys:
                    rows['p_full']=np.asarray(self.p_full[start:start+len(rows['x'])],dtype=np.float16)
                for k in keys:
                    if k not in rows:continue
                    gpu.setdefault(k,[]).append(torch.from_numpy(rows[k]).to(device))
            self._gpu={k:torch.cat(v,0) for k,v in gpu.items()}
            return True
        except torch.cuda.OutOfMemoryError:
            self._gpu={};torch.cuda.empty_cache();return False

    def fetch(self,idx,device,keys=MEMORY_KEYS+('p_full',)):
        """idx [B,K] long, -1 = no legal neighbour.  Returns [B,K,...] tensors; padded
        rows are zeros (callers must use idx>=0 as validity)."""
        idx=idx.long();shape=tuple(idx.shape)
        safe=idx.clamp_min(0).reshape(-1)
        out={}
        if self._gpu:
            ids=safe.to(next(iter(self._gpu.values())).device)
            for k in keys:
                if k in self._gpu:
                    v=self._gpu[k][ids]
                    out[k]=v.reshape(shape+tuple(v.shape[1:])).to(device)
        missing=[k for k in keys if k not in out and (k!='p_full' or self.p_full is not None)]
        if missing:
            host=self.rows(safe.cpu().numpy())
            if 'p_full' in missing:host['p_full']=np.asarray(self.p_full[safe.cpu().numpy()],dtype=np.float16)
            for k in missing:
                v=torch.from_numpy(np.ascontiguousarray(host[k]))
                out[k]=v.reshape(shape+tuple(v.shape[1:])).to(device,non_blocking=True)
        pad=(idx<0).to(device)
        for k,v in out.items():
            m=pad.view(shape+(1,)*(v.dim()-len(shape)))
            out[k]=torch.where(m,torch.zeros_like(v),v)
        return out
