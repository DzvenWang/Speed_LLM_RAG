"""Train-only episode memory; all future values are completed past episodes.

Memory uses CPU arrays and GPU-readable small keys, with sparse target/neighborhood
value gathering. Residual provenance is verified on production starts.
"""
from pathlib import Path
import json,warnings
import numpy as np,torch
from .config import dump_json,hash_file

def _episode_key(x,mask):
    # x/mask [M,H,N]; key solely from historical information
    ok=mask.float();count=ok.sum(1).clamp_min(1)
    mean=(x*ok).sum(1)/count
    var=((x-mean[:,None,:]).square()*ok).sum(1)/count
    trend=(x[:,-1]-x[:,0])*mask[:,-1]*mask[:,0]
    coverage=ok.mean(1)
    return torch.stack([mean,torch.sqrt(var+1e-5),trend,coverage],-1)

class EpisodeMemory:
    def __init__(self,history,mask,future,future_valid,starts,origin,base_forecast=None,metadata=None):
        x=torch.as_tensor(history,dtype=torch.float32,device='cpu');m=torch.as_tensor(mask,dtype=torch.bool,device='cpu')
        y=torch.as_tensor(future,dtype=torch.float32,device='cpu');yv=torch.as_tensor(future_valid,dtype=torch.bool,device='cpu')
        self.x=x;self.mask=m;self.future=y;self.future_valid=yv
        self.starts=torch.as_tensor(starts,dtype=torch.long,device='cpu')
        self.origin=torch.as_tensor(origin,dtype=torch.long,device='cpu')
        if base_forecast is None:self.residual=torch.zeros_like(y);self.residual_valid=torch.zeros_like(yv)
        else:self.residual=y-torch.as_tensor(base_forecast,dtype=torch.float32,device='cpu');self.residual_valid=yv.clone()
        self.metadata=dict(metadata or {})
        self.node_key_cpu=_episode_key(x,m)
        self._node_keys_device={}
    @property
    def n(self):return self.x.shape[-1]
    @property
    def m(self):return self.x.shape[0]
    def keys(self,device):
        k=str(device)
        if k not in self._node_keys_device:self._node_keys_device[k]=self.node_key_cpu.to(device)
        return self._node_keys_device[k]
    def valid_for(self,origins,device,purge=0):
        # memory future end MUST precede current origin, including val/test.
        ends=(self.starts+self.x.shape[1]+self.future.shape[1]-1).to(device)
        return ends[None,:]+int(purge)<=origins[:,None]
    def fetch_node(self,episode_ids,source,device):
        # IDs [B,N,K] -> [B,N,K,P] without materializing [B,N,K,P,N].
        ids=episode_ids.detach().cpu().long().clamp(0,self.m-1)
        b,n,k=ids.shape
        nodes=torch.arange(n).view(1,n,1)
        table=getattr(self,source).permute(0,2,1)
        return table[ids,nodes].to(device,non_blocking=True)
    def fetch_neighbors(self,episode_ids,neighbor_ids,source,device):
        # [B,N,K,L,P], bounded 1-hop graph neighborhood.
        ids=episode_ids.detach().cpu().long().clamp(0,self.m-1)
        neighbors=neighbor_ids.detach().cpu().long();table=getattr(self,source).permute(0,2,1)
        return table[ids[...,None],neighbors[None,:,None,:]].to(device,non_blocking=True)
    def save(self,path):
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(p,x=self.x.numpy().astype('float16'),mask=self.mask.numpy(),future=self.future.numpy().astype('float16'),
                            future_valid=self.future_valid.numpy(),starts=self.starts.numpy(),origin=self.origin.numpy(),
                            residual=self.residual.numpy().astype('float16'),residual_valid=self.residual_valid.numpy())
        dump_json(p.with_suffix('.json'),{'sha256':hash_file(p),'episodes':self.m,'nodes':self.n,**self.metadata})
    @classmethod
    def load(cls,path,expected=None):
        p=Path(path);meta=json.loads(p.with_suffix('.json').read_text())
        if hash_file(p)!=meta['sha256']:raise RuntimeError('Memory SHA mismatch')
        for k,v in (expected or {}).items():
            if meta.get(k)!=v:raise RuntimeError(f'Memory {k} mismatch: {meta.get(k)} vs {v}')
        z=np.load(p,allow_pickle=False)
        obj=cls(z['x'],z['mask'],z['future'],z['future_valid'],z['starts'],z['origin'],metadata=meta)
        obj.residual=torch.as_tensor(np.array(z['residual'],dtype=np.float32,copy=True))
        obj.residual_valid=torch.as_tensor(np.array(z['residual_valid'],dtype=bool,copy=True))
        return obj

@torch.no_grad()
def from_support_cache(cache,backbone,max_episodes,mode='checkpoint_in_sample',external_base_file=None,device='cpu',batch_size=16):
    from .protocol import SupportDataset,make_loader,device_batch
    ds=SupportDataset(cache,'train')
    starts=ds.meta.window_start.to_numpy(np.int64)
    _,uni=np.unique(starts,return_index=True)
    ix=np.sort(uni)
    oos=None;lookup=None
    if mode=='external_oos':
        if not external_base_file:raise RuntimeError('external_oos requires --oos-forecast-file')
        oos=np.load(external_base_file,allow_pickle=False)
        if not all(k in oos for k in ('window_start','prediction','trained_until')):
            raise RuntimeError('OOF file must contain window_start, prediction, trained_until')
        lookup={int(k):i for i,k in enumerate(oos['window_start'])}
        # Chronological OOF warm-up interval can have no eligible producer model.
        ix=np.asarray([i for i in ix if int(starts[i]) in lookup],dtype=np.int64)
        if len(ix)==0:raise RuntimeError('No eligible OOF memory episodes')
    if max_episodes and len(ix)>max_episodes:ix=ix[np.linspace(0,len(ix)-1,max_episodes,dtype=np.int64)]
    arr=ds.arr
    obs=np.asarray(arr['x_obs_norm'][ix],dtype='float32');mask=np.asarray(arr['history_observed'][ix],dtype=bool)
    y=np.asarray(arr['y_norm'][ix],dtype='float32');valid=np.asarray(arr['future_valid'][ix],dtype=bool)
    starts_sel=starts[ix]
    if mode=='external_oos':
        matches=[lookup[int(s)] for s in starts_sel]
        training_cutoff=np.asarray(oos['trained_until'][matches],dtype='int64')
        if np.any(training_cutoff>=starts_sel):
            raise RuntimeError('OOF leakage: a producer model trained through its historical prediction window')
        base=np.asarray(oos['prediction'][matches],dtype='float32')
        if base.shape!=y.shape:raise RuntimeError('OOF prediction shape mismatch')
        if not np.isfinite(base).all():raise RuntimeError('Nonfinite OOF predictions')
        provenance='external_oos'
    elif mode=='checkpoint_in_sample':
        warnings.warn('EXPLORATORY residuals: backbone trained on overlapping train data; not OOF.',RuntimeWarning)
        preds=[]
        for low in range(0,len(ix),batch_size):
            ids=ix[low:low+batch_size]
            x=torch.as_tensor(np.asarray(arr['x_obs_norm'][ids],dtype='float32'),device=device)
            m=torch.as_tensor(np.asarray(arr['history_observed'][ids],dtype=bool),device=device)
            tod=torch.as_tensor(np.asarray(arr['tod_hist'][ids]),device=device)
            dow=torch.as_tensor(np.asarray(arr['dow_hist'][ids]),device=device)
            preds.append(backbone(x,m,tod,dow)['prediction'].cpu())
        base=torch.cat(preds,0).numpy();provenance='checkpoint_in_sample_NOT_OOF'
    else:raise ValueError('residual mode must be external_oos or checkpoint_in_sample')
    memory=EpisodeMemory(obs,mask,y,valid,starts_sel,starts_sel+ds.h-1,base,metadata={'residual_provenance':provenance})
    return memory
