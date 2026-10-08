from __future__ import annotations
import numpy as np, torch
from pathlib import Path
from .config import json_write,sha256

class TrainMemory:
    """Train-only historical window store, no stored or fetched future truth."""
    def __init__(self,x,valid,starts,latent=None,device='cpu',tod=None,dow=None):
        self.x=torch.as_tensor(np.asarray(x),dtype=torch.float32,device=device)
        self.valid=torch.as_tensor(np.asarray(valid),dtype=torch.bool,device=device)
        self.starts=torch.as_tensor(np.asarray(starts),dtype=torch.long,device=device)
        self.latent=(torch.as_tensor(np.asarray(latent),dtype=torch.float32,device=device) if latent is not None else None)
        self.tod=(torch.as_tensor(np.asarray(tod),dtype=torch.long,device=device) if tod is not None else None)
        self.dow=(torch.as_tensor(np.asarray(dow),dtype=torch.long,device=device) if dow is not None else None)
        self.keys=None
        assert self.x.ndim==3 and self.valid.shape==self.x.shape and len(self.starts)==len(self.x)
    def to(self,device):
        for k in ('x','valid','starts','latent','keys','tod','dow'):
            v=getattr(self,k)
            if v is not None:setattr(self,k,v.to(device))
        return self
    def fetch(self,ids):
        safe=ids.clamp(min=0,max=len(self.x)-1)
        return self.x[safe],self.valid[safe]
    @classmethod
    def from_cache(cls,cache,max_rows=None,device='cpu',sampling='head'):
        from .data import SupportSplit
        ds=SupportSplit(cache,'train')
        # De-duplicate multiple sensing views of the same historical window.
        starts=ds.meta.window_start.to_numpy(np.int64)
        _,unique=np.unique(starts,return_index=True)
        ix=np.sort(unique)
        if sampling not in ('head','spread'):
            raise ValueError(f'Unsupported TRAIN memory sampling strategy: {sampling}')
        if max_rows is not None:
            count=min(int(max_rows),len(ix))
            if count < 1:raise ValueError('Memory max_rows must be positive')
            if sampling=='spread':
                # Evenly distribute selected, unique TRAIN windows across
                # the whole TRAIN split; never access validation/test here.
                ix=ix[np.linspace(0,len(ix)-1,count,dtype=np.int64)]
            else:
                ix=ix[:count]
        full=ds.arr['x_full_norm'][ix]
        observed=np.asarray(ds.arr['history_observed'][ix],dtype=bool)
        artificial=np.asarray(ds.arr['artificial_missing'][ix],dtype=bool)
        valid=observed|artificial
        return cls(np.asarray(full,dtype=np.float32),valid,starts[ix],device=device,tod=np.asarray(ds.arr['tod_hist'][ix]),dow=np.asarray(ds.arr['dow_hist'][ix]))
    def save(self,path,metadata=None):
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
        kw={'x':self.x.cpu().numpy().astype(np.float16),'valid':self.valid.cpu().numpy(),'starts':self.starts.cpu().numpy()}
        if self.latent is not None:kw['latent']=self.latent.cpu().numpy().astype(np.float16)
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
        return cls(z['x'],z['valid'],z['starts'],z['latent'] if 'latent' in z else None,device=device,tod=z['tod'] if 'tod' in z else None,dow=z['dow'] if 'dow' in z else None)

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
