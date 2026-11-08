"""TCRR v5 frozen base cache with source- and teacher-bound manifests.

Stores [Q,P,N] fp32 forecasts and [Q,N,D] fp16 frozen features.
Dynamic trainable latent projection is deliberately NOT cached.
"""
from __future__ import annotations
from pathlib import Path
import os,json,hashlib
import numpy as np
import torch
from ..config import file_sha256

def digest_model(module):
    h=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        h.update(name.encode('utf-8'))
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def file_key(p):
    p=Path(p);return dict(path=str(p.resolve()),sha256=file_sha256(p))

class FrozenBaseCache:
    def __init__(self,folder,model,dataset,split,checkpoint,adapter_path,device,
                 build_batch_size=32,skip_build=False):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.forecast_path=self.folder/f'{split}_base_float32.npy'
        self.context_path=self.folder/f'{split}_latent_float16.npy'
        self.manifest_path=self.folder/f'{split}_manifest.json'
        q=int(len(dataset));p=int(model.stla.base.out_steps);n=int(model.stla.base.num_nodes)
        d=int(model.stla.model_dim)
        identity=dict(schema=5,query_count=q,horizon=p,nodes=n,model_dim=d,
                      backbone=file_key(checkpoint),adapter=file_key(adapter_path),
                      split=split,teacher_sha256=digest_model(model.stla),
                      x=file_key(Path(dataset.root)/f'{split}_x_obs_norm.npy'),
                      mask=file_key(Path(dataset.root)/f'{split}_history_observed.npy'),
                      tod=file_key(Path(dataset.root)/f'{split}_tod_hist.npy'),
                      dow=file_key(Path(dataset.root)/f'{split}_dow_hist.npy'))
        valid=False
        if self.manifest_path.is_file() and self.forecast_path.is_file() and self.context_path.is_file():
            try:
                meta=json.loads(self.manifest_path.read_text())
                a=np.load(self.forecast_path,mmap_mode='r');z=np.load(self.context_path,mmap_mode='r')
                valid=meta==identity and a.shape==(q,p,n) and z.shape==(q,n,d)
            except (OSError,ValueError):pass
        if not valid:
            if skip_build:raise RuntimeError('Frozen base cache missing or stale')
            self._build(model,dataset,split,identity,device,build_batch_size)
        self.pred=np.load(self.forecast_path,mmap_mode='r')
        self.lat=np.load(self.context_path,mmap_mode='r')
        print(f'SHARED_BASE_CACHE_PASS split={split} count={q} path={self.folder}',flush=True)
    def _build(self,model,dataset,split,manifest,device,batch_size):
        q=len(dataset);p=manifest['horizon'];n=manifest['nodes'];d=manifest['model_dim']
        ap=self.forecast_path.with_name(self.forecast_path.name+'.partial.npy')
        zp=self.context_path.with_name(self.context_path.name+'.partial.npy')
        out=np.lib.format.open_memmap(ap,mode='w+',dtype='float32',shape=(q,p,n))
        lat=np.lib.format.open_memmap(zp,mode='w+',dtype='float16',shape=(q,n,d))
        model.stla.eval()
        try:
            with torch.inference_mode():
                for s in range(0,q,batch_size):
                    e=min(s+batch_size,q)
                    b=[dataset[i] for i in range(s,e)]
                    x=torch.stack([t['x'] for t in b]).to(device)
                    mask=torch.stack([t['mask'] for t in b]).to(device)
                    tod=torch.stack([t['tod'] for t in b]).to(device)
                    dow=torch.stack([t['dow'] for t in b]).to(device)
                    yp,z=model.stla(x,mask,tod,dow)
                    out[s:e]=yp.float().cpu().numpy()
                    lat[s:e]=z.float().cpu().numpy().astype('float16')
                    if e==q or (s//batch_size)%100==0:
                        print(f'SHARED_BASE_CACHE_PROGRESS split={split} {e}/{q}',flush=True)
            out.flush();lat.flush();del out,lat
            os.replace(ap,self.forecast_path);os.replace(zp,self.context_path)
            tmp=self.manifest_path.with_suffix('.tmp')
            tmp.write_text(json.dumps(manifest,indent=2));os.replace(tmp,self.manifest_path)
        finally:
            if ap.exists():ap.unlink()
            if zp.exists():zp.unlink()
    def fetch(self,indices,device):
        ids=indices.detach().cpu().numpy().astype(np.int64)
        return (torch.from_numpy(np.array(self.pred[ids],copy=True)).to(device,non_blocking=True),
                torch.from_numpy(np.array(self.lat[ids],copy=True)).to(device,non_blocking=True))
