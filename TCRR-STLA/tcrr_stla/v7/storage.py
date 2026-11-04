"""Source-bound FP16/FP32 prediction caches, with atomic completion markers."""
from __future__ import annotations
from pathlib import Path
import os, json, hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..config import file_sha256
from .intervention import fault_partition, teacher_view, decompose_response


def fingerprint_file(p):
    p=Path(p)
    if not p.exists():raise FileNotFoundError(p)
    s=p.stat()
    # Checksums source file bytes, not just mtime. Data changes invalidate cache.
    return {'name':str(p.resolve()),'size':s.st_size,'sha256':file_sha256(p)}


def file_manifest(ds,split,teacher_key,groups):
    files=[ds.root/f'{split}_{name}.npy' for name in ['x_obs_norm','history_observed','tod_hist','dow_hist','future_valid']]
    files.append(ds.root/f'{split}_manifest.csv')
    if split=='train':files += [ds.root/'train_x_full_norm.npy',ds.root/'train_y_norm.npy',
                                      ds.root/'train_artificial_missing.npy']
    return {'schema':7,'teacher_key':teacher_key,'groups':groups,'split':split,'files':[fingerprint_file(p) for p in files],
            'rows':len(ds)}


def _np_open(path,shape,dtype):
    return np.lib.format.open_memmap(str(path), mode='w+',dtype=dtype,shape=shape)


def open_existing_teacher(split,folder,identity,n,nodes,horizon,dim):
    folder=Path(folder);meta=folder/f'{split}_identity.json';complete=folder/f'{split}_DONE'
    if not meta.is_file() or not complete.is_file():return None
    if json.loads(meta.read_text())!=identity:raise RuntimeError(f'Stale teacher identity in {folder}; do not reuse')
    keys=['base','latent']+(['c2','normal','cf'] if split=='train' else [])
    result={}
    for key in keys:
        path=folder/f'{split}_{key}.npy'
        if not path.is_file():raise RuntimeError(f'Incomplete teacher cache {path}')
        a=np.load(path,mmap_mode='r')
        expected=(n,nodes,dim) if key=='latent' else (n,horizon,nodes)
        if a.shape!=expected:raise RuntimeError(f'Teacher cache shape mismatch {path}: {a.shape} != {expected}')
        result[key]=a
    return result

def cache_teacher(split,ds,teacher,folder,device,identity,batch_size=16,workers=0):
    """The only routine which reads x_full_training_only. No val/test CF targets."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    meta=folder/f'{split}_identity.json'; complete=folder/f'{split}_DONE'
    arrays=['base','latent']+(['c2','normal','cf'] if split=='train' else [])
    paths={k:folder/f'{split}_{k}.npy' for k in arrays}
    n=len(ds);p=int(teacher.base.out_steps);nodes=int(teacher.base.num_nodes);dim=int(teacher.model_dim)
    shapes={k:((n,nodes,dim) if k=='latent' else (n,p,nodes)) for k in arrays}
    if meta.is_file() and complete.exists() and all(paths[k].exists() for k in arrays):
        try:
            good=json.loads(meta.read_text())==identity
            good=good and all(np.load(paths[k],mmap_mode='r').shape==shapes[k] for k in arrays)
            if good:
                print(f'V7_TEACHER_CACHE_REUSE split={split} count={n}',flush=True)
                return {k:np.load(paths[k],mmap_mode='r') for k in arrays}
        except (OSError,ValueError):pass
    complete.unlink(missing_ok=True)
    partial={k:Path(str(paths[k])+'.partial.npy') for k in arrays}
    out={k:_np_open(partial[k],shapes[k],np.float16 if k=='latent' else np.float32) for k in arrays}
    teacher.eval()
    loader=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=workers,pin_memory=device.type=='cuda')
    s=0
    try:
        with torch.inference_mode():
            for b in loader:
                x=b['x'].to(device);m=b['mask'].to(device);tod=b['tod'].to(device);dow=b['dow'].to(device)
                yp,z=teacher(x,m,tod,dow)
                e=s+len(x)
                out['base'][s:e]=yp.float().cpu().numpy();out['latent'][s:e]=z.float().cpu().numpy().astype(np.float16)
                if split=='train':
                    xr=b['x_full_training_only'].to(device)
                    source=b['original_valid'].to(device).float()
                    if ((m>.5)&(source<.5)).any():raise RuntimeError('Train source lacks a query-visible value')
                    if ((xr-x).abs()*m).amax().item()>0.01:
                        raise RuntimeError('Train x_full differs from actually observed history')
                    # First create a reference from source-valid measurements only.
                    xr=torch.nan_to_num(xr)*source
                    ref,_=teacher(xr,source,tod,dow)
                    faults,node_masks=fault_partition(m,source,identity['groups'])
                    singles=[]
                    for j in range(faults.shape[1]):
                        xs,ms=teacher_view(xr,source,faults[:,j])
                        py,_=teacher(xs,ms,tod,dow)
                        singles.append(py.float())
                    dec=decompose_response(ref.float(),yp.float(),torch.stack(singles,1),node_masks)
                    for k in ('c2',):
                        out[k][s:e]=dec[k].float().cpu().numpy()
                    out['cf'][s:e]=dec['total'].float().cpu().numpy()
                    out['normal'][s:e]=(b['y_norm'].to(device)-ref.float()).cpu().numpy()
                s=e
                if s==n or s%(batch_size*80)==0:print(f'V7_TEACHER_PROGRESS split={split} {s}/{n}',flush=True)
        for item in out.values():item.flush()
        del out
        for k in arrays:os.replace(partial[k],paths[k])
        tmp=meta.with_suffix('.tmp');tmp.write_text(json.dumps(identity,indent=2));os.replace(tmp,meta)
        complete.write_text('ok\n')
        return {k:np.load(paths[k],mmap_mode='r') for k in arrays}
    finally:
        for pth in partial.values():pth.unlink(missing_ok=True)


class CachedDataset(torch.utils.data.Dataset):
    def __init__(self,source,records):self.source=source;self.records=records
    def __len__(self):return len(self.source)
    def __getitem__(self,i):
        row=self.source[i]
        for k,v in self.records.items():
            row['cached_'+k]=torch.from_numpy(np.array(v[i],dtype=np.float32,copy=True))
        row.pop('x_full_training_only',None)
        return row
