"""Fingerprinted, atomic on-disk caches for ReShift retrieval.

* P_full cache: frozen-STLA forecast of every Train window from its complete
  history, [n,P,N] float16.  Mask independent; built once per S0/backbone.
* Replay bank (one per query set, i.e. Train mask slot / Val severity / Test
  severity): retrieved neighbour ids [Q,K], similarities [Q,K] and the
  failure-replayed forecast P0_c(M_q) [Q,K,P,N] float16.  The replay mask is
  M_q AND the neighbour's own original availability.
Every artefact carries a JSON fingerprint; a mismatch is a hard error.
"""
from __future__ import annotations
import hashlib,json,os,shutil,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from reshift.stla.backbone import base_features
from reshift.retrieval.index import legal_wrong_indices

BANK_SCHEMA=1


def canonical(obj):
    return json.loads(json.dumps(obj,sort_keys=True,allow_nan=False))


def digest(obj,n=16):
    return hashlib.sha256(json.dumps(canonical(obj),sort_keys=True).encode()).hexdigest()[:n]


def code_digest():
    here=Path(__file__).resolve().parent
    h=hashlib.sha256()
    for name in ('bank.py','index.py','memory.py'):
        h.update(name.encode());h.update((here/name).read_bytes())
    h.update((here.parent/'stla'/'backbone.py').read_bytes())
    return h.hexdigest()


def _check_or_claim(folder,fingerprint):
    meta=folder/'fingerprint.json'
    if (folder/'COMPLETE').exists():
        if canonical(json.loads(meta.read_text()))!=canonical(fingerprint):
            raise RuntimeError(f'Retrieval cache fingerprint mismatch in {folder}; delete it or use a new bank root')
        return True
    return False


def _publish(tmp,folder,fingerprint):
    (tmp/'fingerprint.json').write_text(json.dumps(canonical(fingerprint),indent=2,sort_keys=True))
    (tmp/'COMPLETE').write_text(time.strftime('%Y-%m-%dT%H:%M:%S')+'\n')
    if folder.exists():shutil.rmtree(folder)
    os.replace(tmp,folder)


def _lock(folder):
    folder.parent.mkdir(parents=True,exist_ok=True)
    lock=folder.parent/(folder.name+'.lock')
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    except FileExistsError:
        raise RuntimeError(f'Another job is building {folder} (lock {lock}). Wait for it, or delete the stale lock.')
    return lock


@torch.no_grad()
def build_p_full(folder,fingerprint,memory,stla,device,batch=64,log=print):
    folder=Path(folder)
    if not _check_or_claim(folder,fingerprint):
        lock=_lock(folder)
        try:
            tmp=folder.parent/(folder.name+'.partial');shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir(parents=True)
            out=np.lib.format.open_memmap(tmp/'p_full.npy','w+',dtype=np.float16,
                                          shape=(len(memory),memory.horizon,memory.nodes))
            start=time.time();stla.eval()
            for s in range(0,len(memory),int(batch)):
                r=memory.rows(np.arange(s,min(len(memory),s+int(batch))))
                x=torch.from_numpy(r['x']).to(device);m=torch.from_numpy(r['ov']).float().to(device)
                with torch.autocast(device_type=torch.device(device).type,dtype=torch.float16,
                                    enabled=torch.device(device).type=='cuda'):
                    _,_,p=base_features(stla,x,m,torch.from_numpy(r['tod']).to(device),
                                        torch.from_numpy(r['dow']).to(device))
                p=p.float()
                if not torch.isfinite(p).all():raise RuntimeError('non-finite P_full')
                out[s:s+len(x)]=p.cpu().numpy().astype(np.float16)
            out.flush();del out
            _publish(tmp,folder,fingerprint)
            log(f'RESHIFT_P_FULL_PASS rows={len(memory)} seconds={time.time()-start:.1f} path={folder}')
        finally:
            lock.unlink(missing_ok=True)
    return np.load(folder/'p_full.npy',mmap_mode='r')


@torch.no_grad()
def replay(stla,memory,idx,query_mask,device,chunk=64):
    """P0_c(M_q) for idx [B,K]; query_mask [B,L,N].  Padded neighbours -> 0."""
    B,K=idx.shape
    cand=memory.fetch(idx,device,keys=('x','ov','tod','dow'))
    m=query_mask.to(device).float()[:,None]*cand['ov'].float()
    x=cand['x'].float()*m
    L,N=x.shape[-2:]
    flat=lambda t,*tail:t.reshape((B*K,)+tail)
    fx=flat(x,L,N);fm=flat(m,L,N);ft=flat(cand['tod'],L);fd=flat(cand['dow'],L)
    pieces=[]
    for s in range(0,B*K,int(chunk)):
        with torch.autocast(device_type=torch.device(device).type,dtype=torch.float16,
                            enabled=torch.device(device).type=='cuda'):
            _,_,p=base_features(stla,fx[s:s+chunk],fm[s:s+chunk],ft[s:s+chunk],fd[s:s+chunk])
        pieces.append(p.float())
    p0=torch.cat(pieces,0).reshape(B,K,*pieces[0].shape[1:])
    p0=torch.where((idx<0).to(device)[:,:,None,None],torch.zeros_like(p0),p0)
    if not torch.isfinite(p0).all():raise RuntimeError('non-finite replay forecast')
    return p0


@torch.no_grad()
def build_bank(folder,fingerprint,dataset,memory,index,stla,device,top_k,span,embargo,
               min_valid,query_batch=32,replay_chunk=64,workers=0,wrong=False,log=print):
    """One pass over `dataset` (masked inputs only): shortlist, (optionally
    swap in legal wrong cases), replay, write rows at dataset positions."""
    folder=Path(folder)
    if _check_or_claim(folder,fingerprint):return ReplayBank(folder)
    lock=_lock(folder)
    try:
        tmp=folder.parent/(folder.name+'.partial');shutil.rmtree(tmp,ignore_errors=True);tmp.mkdir(parents=True)
        Q=len(dataset);K=int(top_k)
        ids=np.lib.format.open_memmap(tmp/'idx.npy','w+',dtype=np.int32,shape=(Q,K))
        sims=np.lib.format.open_memmap(tmp/'sim.npy','w+',dtype=np.float32,shape=(Q,K))
        p0s=np.lib.format.open_memmap(tmp/'p0.npy','w+',dtype=np.float16,shape=(Q,K,memory.horizon,memory.nodes))
        filled=np.zeros(Q,dtype=bool)
        loader=DataLoader(dataset,batch_size=int(query_batch),shuffle=False,num_workers=int(workers))
        start=time.time();stla.eval();done=0
        for raw in loader:
            rows=raw['index'].numpy().astype(np.int64)
            idx,sim=index.query(raw['x'],raw['mask'],raw['window_start'],K,span,embargo,min_valid)
            if wrong:
                w=legal_wrong_indices(raw['window_start'].numpy(),(idx>=0).cpu().numpy(),
                                      memory.starts,int(span)+int(embargo))
                idx=torch.as_tensor(w,device=idx.device)
            p0=replay(stla,memory,idx,raw['mask'],device,replay_chunk)
            ids[rows]=idx.cpu().numpy().astype(np.int32);sims[rows]=sim.cpu().numpy()
            p0s[rows]=p0.cpu().numpy().astype(np.float16);filled[rows]=True
            done+=len(rows)
            if (done//int(query_batch))%200==0:
                log(f'RESHIFT_BANK_PROGRESS {folder.name} {done}/{Q} elapsed={time.time()-start:.1f}s')
        if not filled.all():raise RuntimeError(f'Bank rows not filled: {int((~filled).sum())}')
        legal=float((np.asarray(ids)>=0).mean())
        for a in (ids,sims,p0s):a.flush()
        del ids,sims,p0s
        _publish(tmp,folder,fingerprint)
        log(f'RESHIFT_BANK_PASS {folder.name} queries={Q} k={K} legal_fraction={legal:.4f} '
            f'wrong={bool(wrong)} seconds={time.time()-start:.1f}')
    finally:
        lock.unlink(missing_ok=True)
    return ReplayBank(folder)


class ReplayBank:
    def __init__(self,folder,on_gpu=False,device='cpu'):
        self.folder=Path(folder)
        if not (self.folder/'COMPLETE').exists():raise RuntimeError(f'Incomplete bank {folder}')
        self.idx=np.load(self.folder/'idx.npy',mmap_mode='r')
        self.sim=np.load(self.folder/'sim.npy',mmap_mode='r')
        self.p0=np.load(self.folder/'p0.npy',mmap_mode='r')
        self.fingerprint=json.loads((self.folder/'fingerprint.json').read_text())
        self.gpu=None
        if on_gpu and torch.device(device).type=='cuda':
            try:
                self.gpu={'idx':torch.from_numpy(np.array(self.idx)).to(device),
                          'sim':torch.from_numpy(np.array(self.sim)).to(device),
                          'p0':torch.from_numpy(np.array(self.p0)).to(device)}
            except torch.cuda.OutOfMemoryError:
                self.gpu=None;torch.cuda.empty_cache()

    def __len__(self):return len(self.idx)

    def fetch(self,rows,device):
        if self.gpu is not None:
            r=rows.long().to(self.gpu['idx'].device)
            return {k:v[r].to(device) for k,v in self.gpu.items()}
        r=rows.detach().cpu().numpy().astype(np.int64)
        if len(r) and (r.min()<0 or r.max()>=len(self.idx)):raise IndexError('bank row out of range')
        return {'idx':torch.from_numpy(np.asarray(self.idx[r],dtype=np.int64)).to(device),
                'sim':torch.from_numpy(np.asarray(self.sim[r],dtype=np.float32)).to(device),
                'p0':torch.from_numpy(np.asarray(self.p0[r])).to(device)}
