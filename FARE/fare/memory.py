from __future__ import annotations
import numpy as np, torch
from pathlib import Path
from .config import json_write,sha256

class TrainMemory:
    """Train-only historical windows, optionally with strictly historical observed outcomes."""
    def __init__(self,x,valid,starts,latent=None,device='cpu',tod=None,dow=None,future=None,future_valid=None):
        self.x=torch.as_tensor(np.asarray(x),dtype=torch.float32,device=device)
        self.valid=torch.as_tensor(np.asarray(valid),dtype=torch.bool,device=device)
        self.starts=torch.as_tensor(np.asarray(starts),dtype=torch.long,device=device)
        self.latent=(torch.as_tensor(np.asarray(latent),dtype=torch.float32,device=device) if latent is not None else None)
        self.tod=(torch.as_tensor(np.asarray(tod),dtype=torch.long,device=device) if tod is not None else None)
        self.dow=(torch.as_tensor(np.asarray(dow),dtype=torch.long,device=device) if dow is not None else None)
        self.future=(torch.as_tensor(np.asarray(future),dtype=torch.float32,device=device) if future is not None else None)
        self.future_valid=(torch.as_tensor(np.asarray(future_valid),dtype=torch.bool,device=device) if future_valid is not None else None)
        self.keys=None
        # Derived inference-only region aggregates; never serialize into checkpoints.
        self.region_future=None
        self.region_future_valid=None
        self._region_cache_signature=None
        assert self.x.ndim==3 and self.valid.shape==self.x.shape and len(self.starts)==len(self.x)
    def to(self,device):
        for k in ('x','valid','starts','latent','keys','tod','dow','future','future_valid','region_future','region_future_valid'):
            v=getattr(self,k)
            if v is not None:setattr(self,k,v.to(device))
        return self
    def fetch(self,ids):
        safe=ids.clamp(min=0,max=len(self.x)-1)
        return self.x[safe],self.valid[safe]

    def fetch_nodes(self, ids, node_ids, kind='history', time_index=None):
        """Indexed gather without materializing [B,G,K,T,N].

        ids: [B,G,K], node_ids: [G,S] including padded region entries.
        Returns [B,G,K,T,S] or [B,G,K,S] when time_index is specified.
        """
        if ids.ndim != 3 or node_ids.ndim != 2 or ids.shape[1] != node_ids.shape[0]:
            raise ValueError('fetch_nodes expects ids [B,G,K], nodes [G,S]')
        if kind == 'history':
            source, validity = self.x, self.valid
        elif kind == 'future':
            if self.future is None or self.future_valid is None:
                raise ValueError('Future evidence missing from memory')
            source, validity = self.future, self.future_valid
        else:
            raise ValueError(f'Unsupported evidence kind: {kind}')
        safe = ids.clamp(min=0, max=len(source)-1)
        nodes = node_ids.clamp(min=0, max=source.shape[-1]-1)
        # Advanced tensor indexing gathers just the required region/anchor nodes.
        row = safe[..., None, None]
        node = nodes[None, :, None, None, :]
        if time_index is None:
            clock = torch.arange(source.shape[1], device=source.device)[None, None, None, :, None]
            return source[row, clock, node], validity[row, clock, node]
        return (source[safe[..., None], time_index, nodes[None, :, None, :]],
                validity[safe[..., None], time_index, nodes[None, :, None, :]])

    def fetch_target(self, ids, kind='history', node_ids=None):
        """Node-specific sparse gather: [B,N,K] IDs -> [B,N,K,T]."""
        if ids.ndim != 3:
            raise ValueError('Target candidate ids must be [B,N,K]')
        if node_ids is None:
            if ids.shape[1] != self.x.shape[-1]:
                raise ValueError('Full target ids must include every graph node')
            node_ids = torch.arange(self.x.shape[-1], device=ids.device)
        else:
            node_ids = torch.as_tensor(node_ids, dtype=torch.long, device=ids.device)
            if node_ids.ndim != 1 or ids.shape[1] != len(node_ids):
                raise ValueError('node_ids must match target candidate node dimension')
            if bool(((node_ids < 0) | (node_ids >= self.x.shape[-1])).any()):
                raise IndexError('Invalid audit target node index')
        if kind == 'history':
            x, valid = self.x, self.valid
        elif kind == 'future':
            if self.future is None or self.future_valid is None:
                raise RuntimeError('Train-only future memory is required')
            x, valid = self.future, self.future_valid
        else:
            raise ValueError(f'Unknown evidence type: {kind}')
        b,n,k=ids.shape
        nodes=node_ids[None,:,None]
        ids=ids.clamp(0,len(x)-1)
        # [M,T,N] -> [M,N,T], avoiding an intermediate full-network candidate.
        return x.permute(0,2,1)[ids,nodes],valid.permute(0,2,1)[ids,nodes]

    def fetch_target_latent(self,ids,dim,node_ids=None):
        if self.latent is None: raise RuntimeError('Historical STLA latent cache not built')
        if ids.ndim!=3:
            raise ValueError('Latent target candidates must be [B,N,K]')
        if node_ids is None:
            if ids.shape[1]!=self.latent.shape[1]:
                raise ValueError('Full latent ids must cover all nodes')
            node_ids=torch.arange(self.latent.shape[1],device=ids.device)
        else:
            node_ids=torch.as_tensor(node_ids,dtype=torch.long,device=ids.device)
            if node_ids.ndim!=1 or ids.shape[1]!=len(node_ids):
                raise ValueError('Latent node_ids shape mismatch')
            if bool(((node_ids<0)|(node_ids>=self.latent.shape[1])).any()):
                raise IndexError('Invalid latent audit target node index')
        nodes=node_ids[None,:,None]
        z=self.latent[ids.clamp(0,len(self.latent)-1),nodes].float()
        if z.shape[-1]<dim:
            return torch.nn.functional.pad(z,(0,dim-z.shape[-1]))
        return z[...,:dim]

    @torch.no_grad()
    def ensure_region_future(self, regions, chunk_size=512):
        """Cache exact valid-only region means of historical futures, once per memory."""
        if self.future is None or self.future_valid is None:
            return
        signature=(self.future.data_ptr(),self.future_valid.data_ptr(),
                   regions.target_mask.data_ptr(),regions.g,regions.n)
        if self._region_cache_signature==signature and self.region_future is not None:
            return
        weights=regions.target_mask.to(device=self.future.device, dtype=self.future.dtype)
        means=[];counts=[]
        for lo in range(0,len(self.future),chunk_size):
            x=self.future[lo:lo+chunk_size]
            valid=self.future_valid[lo:lo+chunk_size]
            count=torch.einsum('mpn,gn->mgp',valid.to(x.dtype),weights)
            num=torch.einsum('mpn,gn->mgp',x*valid.to(x.dtype),weights)
            means.append(num/count.clamp_min(1))
            counts.append(count > 0)
        self.region_future=torch.cat(means,dim=0)
        self.region_future_valid=torch.cat(counts,dim=0)
        self._region_cache_signature=signature
    @classmethod
    def from_cache(cls,cache,max_rows=None,device='cpu',sampling='head'):
        from .data import SupportSplit
        ds=SupportSplit(cache,'train')
        # De-duplicate multiple sensing views of the same historical window.
        starts=ds.meta.window_start.to_numpy(np.int64)
        _,unique=np.unique(starts,return_index=True)
        ix=np.sort(unique)
        if sampling not in ('head','spread','stratified_temporal'):
            raise ValueError(f'Unsupported TRAIN memory sampling strategy: {sampling}')
        if max_rows is not None:
            count=min(int(max_rows),len(ix))
            if count < 1:raise ValueError('Memory max_rows must be positive')
            if sampling in ('spread','stratified_temporal'):
                # Evenly distribute selected, unique TRAIN windows across
                # the whole TRAIN split; never access validation/test here.
                ix=ix[np.linspace(0,len(ix)-1,count,dtype=np.int64)]
            else:
                ix=ix[:count]
        full=ds.arr['x_full_norm'][ix]
        observed=np.asarray(ds.arr['history_observed'][ix],dtype=bool)
        artificial=np.asarray(ds.arr['artificial_missing'][ix],dtype=bool)
        valid=observed|artificial
        return cls(np.asarray(full,dtype=np.float32),valid,starts[ix],device=device,tod=np.asarray(ds.arr['tod_hist'][ix]),dow=np.asarray(ds.arr['dow_hist'][ix]),future=(np.asarray(ds.arr['y_norm'][ix],dtype=np.float32) if 'y_norm' in ds.arr else None),future_valid=(np.asarray(ds.arr['future_valid'][ix],dtype=bool) if 'future_valid' in ds.arr else None))
    def save(self,path,metadata=None):
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
        kw={'x':self.x.cpu().numpy().astype(np.float16),'valid':self.valid.cpu().numpy(),'starts':self.starts.cpu().numpy()}
        if self.latent is not None:kw['latent']=self.latent.cpu().numpy().astype(np.float16)
        if self.future is not None:kw['future']=self.future.cpu().numpy().astype(np.float16)
        if self.future_valid is not None:kw['future_valid']=self.future_valid.cpu().numpy()
        if self.tod is not None:kw['tod']=self.tod.cpu().numpy().astype(np.int16)
        if self.dow is not None:kw['dow']=self.dow.cpu().numpy().astype(np.int16)
        np.savez_compressed(p,**kw)
        json_write(p.with_suffix('.json'),{'rows':len(self.x),'nodes':self.x.shape[-1],'sha256':sha256(p),**(metadata or {})})
    @classmethod
    def load(cls,path,device='cpu'):
        p=Path(path);z=np.load(p,allow_pickle=False)
        import json
        mp=p.with_suffix('.json')
        if mp.exists():
            meta=json.loads(mp.read_text(encoding='utf-8'))
            if meta.get('sha256')!=sha256(p):raise RuntimeError('Retrieval memory checksum mismatch')
        return cls(z['x'],z['valid'],z['starts'],z['latent'] if 'latent' in z else None,device=device,tod=z['tod'] if 'tod' in z else None,dow=z['dow'] if 'dow' in z else None,future=z['future'] if 'future' in z else None,future_valid=z['future_valid'] if 'future_valid' in z else None)

@torch.no_grad()
def populate_latents(memory,model,batch_size=32,dim=32):
    if memory.latent is not None:return memory
    if memory.tod is None or memory.dow is None:raise RuntimeError('Memory must store TOD/DOW for teacher latent cache')
    model.eval();parts=[]
    for lo in range(0,len(memory.x),batch_size):
        hi=min(lo+batch_size,len(memory.x))
        h=model.teacher_hidden(memory.x[lo:hi].float(),memory.valid[lo:hi],memory.tod[lo:hi],memory.dow[lo:hi])
        parts.append(h.mean(1)[...,:dim].detach().to(dtype=torch.float16).cpu())
    memory.latent=torch.cat(parts,0).to(memory.x.device)
    return memory
