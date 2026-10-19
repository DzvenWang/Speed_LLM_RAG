"""FARE 5.0.1 episode store: completed historical evidence, compact keys, memmapped files.

A historical future may be consulted only after it has actually ended.  Strict
Formal residuals require separately produced, verified OOS predictions.
"""
from __future__ import annotations
from pathlib import Path
import json,warnings
import numpy as np,torch
from .config import dump_json,hash_file


def _episode_key(x,mask):
    m=mask.float();den=m.sum(1).clamp_min(1)
    mean=(x*m).sum(1)/den
    std=torch.sqrt(((x-mean[:,None])**2*m).sum(1)/den+1e-5)
    trend=(x[:,-1]-x[:,0])*mask[:,-1]*mask[:,0]
    return torch.stack([mean,std,trend,m.mean(1)],-1)


def _capability_key(future,valid,residual,residual_valid,history,mask):
    # Normalized, per-node capability from *past completed* historical evidence.
    # Summaries are descriptive, not supervised physical event labels.
    mv=valid.float();rv=residual_valid.float()
    last_index=torch.where(mask,torch.arange(mask.shape[1])[None,:,None],-1).amax(1)
    h=torch.gather(history,1,last_index.clamp_min(0)[:,None,:]).squeeze(1)
    evolution=(future-h[:,None])*mv
    r=residual*rv
    def mean(v,w):return (v*w).sum(1)/w.sum(1).clamp_min(1)
    return torch.stack([mean(evolution,mv),mean(evolution.abs(),mv),
                        mean(r,rv),mean(r.abs(),rv),
                        (evolution[:,-1]-evolution[:,0])*mv[:,-1]*mv[:,0],
                        mean((evolution[:,1:]-evolution[:,:-1]).abs(),mv[:,1:]*mv[:,:-1]),
                        mv.mean(1),rv.mean(1)],-1)


class EpisodeMemory:
    def __init__(self,history,mask,future,future_valid,starts,origin,base_forecast=None,failure_delta=None,metadata=None,episode_info=None):
        def cpu(x,dtype):
            if torch.is_tensor(x):return x.detach().to(device='cpu',dtype=dtype).clone()
            return torch.as_tensor(np.asarray(x).copy(),dtype=dtype,device='cpu')
        self.x=cpu(history,torch.float32);self.mask=cpu(mask,torch.bool)
        self.future=cpu(future,torch.float32);self.future_valid=cpu(future_valid,torch.bool)
        self.starts=cpu(starts,torch.long);self.origin=cpu(origin,torch.long)
        if base_forecast is None:
            self.residual=torch.zeros_like(self.future);self.residual_valid=torch.zeros_like(self.future_valid)
        else:
            self.residual=self.future-cpu(base_forecast,torch.float32)
            self.residual_valid=self.future_valid.clone()
        if failure_delta is None:
            self.failure_delta=torch.zeros_like(self.future)
            self.failure_valid=torch.zeros_like(self.future_valid)
        else:
            self.failure_delta=cpu(failure_delta,torch.float32)
            self.failure_valid=self.future_valid.clone()
        self.metadata=dict(metadata or {})
        self.episode_info={k:(np.asarray(v).astype(str) if np.asarray(v).dtype.kind in ("O","U","S") else np.asarray(v).copy())
                           for k,v in (episode_info or {}).items()}
        if any(len(v)!=len(self.starts) for v in self.episode_info.values()):raise ValueError("Episode info length mismatch")
        self.node_key_cpu=_episode_key(self.x,self.mask)
        self.capability_cpu=_capability_key(self.future,self.future_valid,self.residual,
                                             self.residual_valid,self.x,self.mask)
        self.region_key_cpu=None;self.graph_hash=None
        self._device_keys={}
    @property
    def m(self):return len(self.starts)
    @property
    def n(self):return self.x.shape[-1]
    def attach_graph(self,graph):
        owner=graph.region_membership.detach().cpu()
        self.region_key_cpu=torch.einsum('gn,mnd->mgd',owner,self.node_key_cpu)
        self.region_capability_cpu=torch.einsum('gn,mnd->mgd',owner,self.capability_cpu)
        self._device_keys.clear()
    def keys(self,device):
        # Compact CPU/GPU features; never move all per-node historical keys to GPU.
        return self.node_key_cpu.to(device)
    def region_keys(self,device):
        if self.region_key_cpu is None:raise RuntimeError('attach_graph required before retrieval')
        key=str(device)
        if key not in self._device_keys:
            self._device_keys[key]=(self.region_key_cpu.to(device),self.region_capability_cpu.to(device))
        return self._device_keys[key]
    def valid_for(self,origins,device,purge=0):
        ends=self.starts.to(device)+self.x.shape[1]+self.future.shape[1]-1
        return ends[None]+int(purge)<=origins[:,None]
    def fetch(self,ids,nodes,source,device):
        """Fetch arbitrary [B,N,K,L] node IDs, without moving full Memory to GPU."""
        ep=ids.detach().to('cpu',dtype=torch.long).clamp(0,self.m-1)
        src=nodes.detach().to('cpu',dtype=torch.long).clamp(0,self.n-1)
        table=getattr(self,source)
        if table.ndim!=3:raise ValueError('fetch source must be a temporal [M,T,N] table')
        data=table.permute(0,2,1)[ep,src]
        return data.to(device,non_blocking=True)
    def fetch_node(self,ids,source,device):
        return self.fetch(ids,torch.arange(ids.shape[1],device=ids.device)[None,:,None],source,device)
    def fetch_paths(self,ids,path_ids,source,device):
        return self.fetch(ids[:,:,:,None],path_ids[None,:,None,:],source,device)
    def fetch_neighbors(self,ids,neighbor_ids,source,device):
        return self.fetch_paths(ids,neighbor_ids,source,device)
    def fetch_node_keys(self,ids,device,source='node_key_cpu'):
        idx=ids.detach().cpu().long().clamp(0,self.m-1)
        nn=torch.arange(ids.shape[1],dtype=torch.long)[None,:,None]
        return getattr(self,source)[idx,nn].to(device,non_blocking=True)
    def save(self,path):
        """Directory of contiguous .npy files; efficient random access and no ZIP inflation."""
        root=Path(path)
        if root.suffix=='.npz':root=root.with_suffix('')
        root.mkdir(parents=True,exist_ok=True)
        names=('x','mask','future','future_valid','starts','origin','residual','residual_valid','failure_delta','failure_valid')
        for name in names:
            value=getattr(self,name).detach().cpu().numpy()
            if name in ('x','future','residual','failure_delta'):value=value.astype('float16')
            np.save(root/f'{name}.npy',value,allow_pickle=False)
        for name,value in self.episode_info.items():
            if name not in {'row_id','episode_key','family','severity','fold_id','distribution','view_id'}:
                raise ValueError(f'Unexpected episode-info field {name}')
            np.save(root/f'info_{name}.npy',value,allow_pickle=False)
        dump_json(root/'manifest.json',{'format':'fare5_memmap','schema_version':6,'episodes':self.m,'nodes':self.n,
                                        'fields':list(names),'episode_info_fields':list(self.episode_info),**self.metadata})
    @classmethod
    def load(cls,path,expected=None):
        root=Path(path)
        if root.suffix=='.npz':root=root.with_suffix('')
        meta=json.loads((root/'manifest.json').read_text())
        if meta.get('schema_version')!=6:raise RuntimeError('FARE 5.0.1 requires schema-6 paired-OOF Memory; legacy Memory cannot be used')
        for k,v in (expected or {}).items():
            if meta.get(k)!=v:raise RuntimeError(f'Memory provenance mismatch {k}: {meta.get(k)} vs {v}')
        def get(name):return np.load(root/f'{name}.npy',mmap_mode='r',allow_pickle=False)
        obj=cls(get('x'),get('mask'),get('future'),get('future_valid'),get('starts'),get('origin'),metadata=meta,
                episode_info={name:np.load(root/f'info_{name}.npy',allow_pickle=False) for name in meta.get('episode_info_fields',[])})
        obj.residual=torch.as_tensor(np.asarray(get('residual'),dtype=np.float32).copy())
        obj.failure_delta=torch.as_tensor(np.asarray(get('failure_delta'),dtype=np.float32).copy())
        obj.failure_valid=torch.as_tensor(np.asarray(get('failure_valid'),dtype=bool).copy())
        obj.residual_valid=torch.as_tensor(np.asarray(get('residual_valid'),dtype=bool).copy())
        obj.capability_cpu=_capability_key(obj.future,obj.future_valid,obj.residual,obj.residual_valid,obj.x,obj.mask)
        return obj


@torch.no_grad()
def from_support_cache(cache,backbone,max_episodes,mode='checkpoint_in_sample',external_base_file=None,
                       device='cpu',batch_size=16):
    from .protocol import SupportDataset
    ds=SupportDataset(cache,'train')
    starts=ds.meta.window_start.to_numpy(np.int64)
    ix=np.arange(len(ds.meta),dtype=np.int64)  # NEVER deduplicate by window_start
    src=None;lookup=None
    if mode=='external_oos_paired':
        if not external_base_file:raise RuntimeError('Formal requires verified OOF paired predictions')
        from .oof import verify_paired
        src=verify_paired(external_base_file,ds,strict=True)
        lookup={int(r):j for j,r in enumerate(src['row_id'])}
        ix=np.asarray([i for i in ix if int(i) in lookup],dtype=np.int64)
        if not len(ix):raise RuntimeError('No OOF episodes match Train cache')
    if max_episodes and len(ix)>max_episodes:
        ix=ix[np.linspace(0,len(ix)-1,int(max_episodes),dtype=np.int64)]
    a=ds.arr
    x=np.asarray(a['x_obs_norm'][ix],dtype=np.float32)
    mask=np.asarray(a['history_observed'][ix],dtype=bool)
    y=np.asarray(a['y_norm'][ix],dtype=np.float32)
    valid=np.asarray(a['future_valid'][ix],dtype=bool)
    selected=starts[ix]
    if mode=='external_oos_paired':
        match=[lookup[int(i)] for i in ix]
        cutoff=np.asarray(src['trained_until'][match],dtype=np.int64)
        # training until MUST end before the episode historical past begins;
        # not sufficient to compare the producer training window START.
        if np.any(cutoff>=selected):raise RuntimeError('Leakage in OOF producer label end times')
        prediction=np.asarray(src['prediction'][match],dtype=np.float32)
        complete_prediction=np.asarray(src['prediction_complete'][match],dtype=np.float32)
        if prediction.shape!=y.shape or complete_prediction.shape!=y.shape or not np.isfinite(prediction).all() or not np.isfinite(complete_prediction).all():
            raise RuntimeError('OOS prediction tensor incompatible')
        failure_delta=complete_prediction-prediction
        provenance='external_oos_paired'
    elif mode=='checkpoint_in_sample':
        warnings.warn('EXPLORATORY ONLY: In-sample residuals are not OOS/OOF',RuntimeWarning)
        prediction=np.empty_like(y)
        for i in range(0,len(ix),batch_size):
            idx=ix[i:i+batch_size]
            tx=lambda name,dt:torch.as_tensor(np.asarray(a[name][idx]),device=device,dtype=dt)
            out=backbone(tx('x_obs_norm',torch.float32),tx('history_observed',torch.bool),
                         tx('tod_hist',torch.long),tx('dow_hist',torch.long))
            prediction[i:i+len(idx)]=out['prediction'].detach().cpu().numpy()
        failure_delta=None
        provenance='checkpoint_in_sample_NOT_OOF'
    else:raise ValueError(f'Unknown residual mode: {mode}')
    info={'row_id':ix,'family':ds.meta['family'].astype(str).to_numpy()[ix] if 'family' in ds.meta else np.full(len(ix),'unknown'),
          'severity':ds.meta['severity'].to_numpy()[ix] if 'severity' in ds.meta else np.full(len(ix),-1)}
    if mode=='external_oos_paired':
        info['episode_key']=np.asarray(src['episode_key'])[match]
        info['fold_id']=np.asarray(src['fold_id'])[match]
    return EpisodeMemory(x,mask,y,valid,selected,selected+ds.h-1,prediction,failure_delta,
                         metadata={'residual_provenance':provenance,'history':ds.h,'horizon':ds.p,
                                   'source_rows':len(ds.meta),'kept_rows':len(ix),
                                   'oof_schema_version':6 if src is not None else None},episode_info=info)
