"""Train-only reference targets and official-observation validation interfaces.

No reference or counterfactual target is passed to the Student.forward method.
"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from ..data import SupportSplit
from .teacher import SCHEMA

class DirectViewDataset(Dataset):
    def __init__(self,cache,root,split='train',limit=None,pair=False,train_fraction=.90,
                 heldout=False,views=None):
        self.ds=SupportSplit(cache,split,limit,cf_training=(split=='train'))
        self.split=split;self.root=Path(root);self.pair=bool(pair)
        self.path=self.root/split
        import json
        self.meta=json.loads((self.path/'manifest.json').read_text())
        if self.meta['schema']!=SCHEMA:raise RuntimeError('v9.3 teacher cache missing/mismatched')
        if self.meta['rows']!=len(self.ds):raise RuntimeError('Teacher and SupportSplit length mismatch')
        self.views=int(self.meta['views']) if views is None else int(views)
        if self.views>self.meta['views']:raise ValueError('More views requested than prepared')
        if self.pair and self.views<2:raise ValueError('Paired training requires multiple views')
        self.arr={name:np.load(self.path/(name+'.npy'),mmap_mode='r',allow_pickle=False)
                  for name in ('base','context','visible','family','severity')}
        if split=='train':
            self.arr.update({name:np.load(self.path/(name+'.npy'),mmap_mode='r',allow_pickle=False)
                             for name in ('reference_base','reference_residual','cf_response','error_damage')})
            starts=self.ds.window_starts[:len(self.ds)]
            order=np.argsort(starts,kind='stable')
            split_at=max(1,min(len(order)-1,int(len(order)*float(train_fraction))))
            boundary=int(starts[order[split_at]])
            gap=self.ds.x_obs.shape[1]+self.ds.y_norm.shape[1]
            if heldout:self.indices=np.flatnonzero(starts>=boundary)
            else:self.indices=np.flatnonzero(starts+gap<boundary)
            if len(self.indices)==0:raise RuntimeError('Chronologically purged TRAIN split empty')
        else:
            self.indices=np.arange(len(self.ds),dtype=np.int64)
            if self.views!=1:raise ValueError('Official validation has only view 0')
    def __len__(self):
        return len(self.indices) if self.pair else len(self.indices)*self.views
    def _one(self,qi,j):
        b=self.ds[int(qi)]
        vis=np.array(self.arr['visible'][qi,j],dtype=np.float32,copy=True)
        if self.split=='train':
            source=np.array(self.ds.x_full[qi],dtype=np.float32,copy=True)
            x=np.where(vis.astype(bool),np.nan_to_num(source,nan=0,posinf=0,neginf=0),0.)
        else:
            source=np.array(self.ds.x_obs[qi],dtype=np.float32,copy=True)
            x=np.where(vis.astype(bool),np.nan_to_num(source,nan=0,posinf=0,neginf=0),0.)
        student=dict(x_obs=torch.from_numpy(x.astype(np.float32)),
                     observed=torch.from_numpy(vis),
                     tod=b['tod'],dow=b['dow'],
                     base=torch.from_numpy(np.array(self.arr['base'][qi,j],dtype=np.float32,copy=True)),
                     context=torch.from_numpy(np.array(self.arr['context'][qi,j],dtype=np.float32,copy=True)))
        data={'student':student,'y':b['y_norm'],'future_valid':b['future_valid'],
              'direct_mask':b['direct_mask'],'query_id':torch.tensor(int(qi)),
              'view':torch.tensor(int(j)),
              'family':torch.tensor(int(self.arr['family'][qi,j])),
              'severity':torch.tensor(int(self.arr['severity'][qi,j]))}
        if self.split=='train':
            data['rref']=torch.from_numpy(np.array(self.arr['reference_residual'][qi],dtype=np.float32,copy=True))
            data['cf_teacher']=torch.from_numpy(np.array(self.arr['cf_response'][qi,j],dtype=np.float32,copy=True))
            data['damage_teacher']=torch.from_numpy(np.array(self.arr['error_damage'][qi,j],dtype=np.float32,copy=True))
        return data
    def __getitem__(self,index):
        if self.pair:
            qi=int(self.indices[index]);a=int(qi%self.views);b=(a+1)%self.views
            return {'a':self._one(qi,a),'b':self._one(qi,b)}
        local,j=divmod(int(index),self.views)
        return self._one(int(self.indices[local]),j)
