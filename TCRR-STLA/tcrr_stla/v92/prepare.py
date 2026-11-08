"""Build deterministic official and paired view evidence with crash-safe memmaps.

Each view is indexed by (query_id,view_id) and retrieval candidates are time legal.
The teacher is frozen; all full-query data use is confined to preparation/train.
"""
from __future__ import annotations
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from ..data import SupportSplit,RetrievalCache
from ..evidence import FutureMemory,counterfactual_residuals
from ..v9.views import paired_mask,mask_sha,FAMILIES
from .retrieval import VisibleCandidateRetriever

PREPARATION_SCHEMA = 92


def _atomic_json(path,data):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,indent=2))
    os.replace(tmp,path)

def _resume_memmaps(folder,specs,manifest_name='progress.json'):
    """Restart safely from last flushed complete chunk. Failed writes never count."""
    folder=Path(folder);p=folder/manifest_name
    pending=[folder/(name+'.partial.npy') for name in specs]
    if p.exists():
        if not all(x.exists() for x in pending):
            raise RuntimeError(f'Partial v9 bank missing arrays: {folder}')
        progress=json.loads(p.read_text())
        shape_signature={name:list(shape) for name,(_,shape) in specs.items()}
        if progress.get('shapes')!=shape_signature:
            raise RuntimeError(f'Partial v9 bank specification mismatch: {folder}')
        arrays={name:np.lib.format.open_memmap(folder/(name+'.partial.npy'),mode='r+') for name in specs}
        for name,(dtype,shape) in specs.items():
            if arrays[name].shape!=shape or arrays[name].dtype!=np.dtype(dtype):
                raise RuntimeError(f'Partial array dtype/shape mismatch: {name}')
        return arrays,int(progress['next_row']),p
    if any(x.exists() for x in pending):
        raise RuntimeError(f'Orphan partial v9 arrays without progress checkpoint in {folder}')
    arrays={name:np.lib.format.open_memmap(folder/(name+'.partial.npy'),mode='w+',
               dtype=dtype,shape=shape) for name,(dtype,shape) in specs.items()}
    _atomic_json(p,{'next_row':0,'shapes':{name:list(shape) for name,(_,shape) in specs.items()}})
    return arrays,0,p

def _checkpoint_arrays(arrays,p,next_row):
    for arr in arrays.values():arr.flush()
    old=json.loads(p.read_text());old['next_row']=int(next_row)
    _atomic_json(p,old)

def source_mask(ds,ids):
    observed=np.asarray(ds.observed[ids],dtype=bool)
    artificial=np.asarray(ds.original_valid[ids],dtype=bool) if ds.original_valid is not None else np.zeros_like(observed)
    return observed|artificial

def time_legal_ids(query_start,train_starts,k,history,horizon):
    """Time-only retrieval is legal for alternative paired views without masked-input leakage.
    The train candidate's full input+future interval ends before query origin.
    """
    cutoff=int(query_start)-(int(history)+int(horizon))
    legal=np.flatnonzero(train_starts<=cutoff)
    legal=legal[np.argsort(train_starts[legal],kind='stable')]
    if len(legal)==0:return np.zeros(k,dtype=np.int32)
    # Recency plus spread: stratified quantiles of past, unique if feasible.
    pos=np.linspace(len(legal)-1,max(0,len(legal)-1-12*k),k).round().astype(int)
    chosen=legal[pos]
    if len(set(chosen))<k:
        chosen=legal[max(0,len(legal)-k):][::-1]
    if len(chosen)<k:chosen=np.pad(chosen,(0,k-len(chosen)),mode='edge')
    return chosen.astype(np.int32)

def _meta(name,dataset,view,teacher_hash,short_hash,shape):
    return {'schema':9,'name':name,'split':dataset.split,'view_id':view,
            'rows':len(dataset),'shape':list(shape),'teacher':teacher_hash,'shortlist':short_hash}

def _paths(out,split,view):return out/f'{split}_view{view}'

class PreparedView:
    def __init__(self,folder):
        folder=Path(folder);self.path=folder
        meta=json.loads((folder/'manifest.json').read_text())
        if meta['schema']!=PREPARATION_SCHEMA:raise ValueError('wrong v9.1 cache schema')
        self.meta=meta
        self.arrays={name:np.load(folder/(name+'.npy'),mmap_mode='r',allow_pickle=False)
                     for name in ('base','context','cf','candidate_ids','scores','valid','match','mask','x','teacher_cf','family','severity','mask_hash')}
        rows=meta['rows'];k=meta['k'];h=meta['h'];n=meta['n']
        assert self.arrays['cf'].shape==(rows,k,h,n)
        for key in self.arrays:
            if self.arrays[key].shape[0]!=rows:raise ValueError(f'{key} row mismatch')

class V9EvidenceDataset(torch.utils.data.Dataset):
    def __init__(self,cache,split,views=(0,),limit=None,root=None,split_key=None):
        if root is None:raise ValueError('V9 evidence root must be explicit')
        self.views={v:PreparedView(_paths(Path(root),split_key or split,v)) for v in views}
        self.ds=SupportSplit(cache,split,min(limit or 10**12, self.views[views[0]].meta['rows']))
        self.vnames=tuple(views)
        self.normal=np.load(Path(root)/'normal_train.npy',mmap_mode='r')
        self.num=len(self.ds)
        for v in self.views.values():
            if len(self.ds)!=v.meta['rows']:
                raise RuntimeError('Dataset and prepared view row counts differ')
    def __len__(self):return self.num*len(self.vnames)
    def __getitem__(self,ix):
        qi=int(ix)//len(self.vnames);vi=self.vnames[int(ix)%len(self.vnames)]
        v=self.views[vi];ids=np.array(v.arrays['candidate_ids'][qi],dtype=np.int64,copy=True)
        raw=self.ds[qi]
        if ids.min()<0 or ids.max()>=len(self.normal):raise IndexError('candidate outside train memory')
        out=dict(query_id=torch.tensor(qi),view_id=torch.tensor(vi),
                 base=torch.from_numpy(np.array(v.arrays['base'][qi],dtype=np.float32,copy=True)),
                 context=torch.from_numpy(np.array(v.arrays['context'][qi],dtype=np.float32,copy=True)),
                 normal=torch.from_numpy(np.array(self.normal[ids],dtype=np.float32,copy=True)),
                 cf=torch.from_numpy(np.array(v.arrays['cf'][qi],dtype=np.float32,copy=True)),
                 match=torch.from_numpy(np.array(v.arrays['match'][qi],dtype=np.float32,copy=True)),
                 candidate_ids=torch.from_numpy(ids.copy()),
                 score=torch.from_numpy(np.array(v.arrays['scores'][qi],dtype=np.float32,copy=True)),
                 valid=torch.from_numpy(np.array(v.arrays['valid'][qi],dtype=np.bool_,copy=True)),
                 x=torch.from_numpy(np.array(v.arrays['x'][qi],dtype=np.float32,copy=True)),
                 teacher_cf_training_only=torch.from_numpy(np.array(v.arrays['teacher_cf'][qi],dtype=np.float32,copy=True)) if self.ds.split=='train' else torch.empty(0),
                 mask=torch.from_numpy(np.array(v.arrays['mask'][qi],dtype=np.float32,copy=True)),
                 y_norm=raw['y_norm'],future_valid=raw['future_valid'],
                 direct_mask=raw['direct_mask'],
                 y_raw=raw['y_raw'],window_start=raw['window_start'])
        return out

@torch.inference_mode()
def prepare_normal(cache,teacher,device,out,chunk=64,limit=None):
    ds=SupportSplit(cache,'train',limit,cf_training=True)
    p=Path(out)/'normal_train.npy'
    if p.exists():
        a=np.load(p,mmap_mode='r',allow_pickle=False)
        if a.shape==(len(ds),*ds.y_norm.shape[1:]):return p
        raise RuntimeError('Existing normal bank shape mismatch: invalidate v9 evidence cache')
    specs={'normal':('float32',(len(ds),*ds.y_norm.shape[1:]))}
    arrays,start,progress=_resume_memmaps(Path(out),specs,manifest_name='normal_progress.json')
    y=arrays['normal']
    for s in range(start,len(ds),chunk):
        e=min(s+chunk,len(ds));ids=np.arange(s,e)
        x=torch.from_numpy(np.nan_to_num(np.array(ds.x_full[ids],np.float32,copy=True))).to(device)
        m=torch.from_numpy(source_mask(ds,ids).astype('float32')).to(device)
        tod=torch.from_numpy(np.array(ds.tod[ids],dtype=np.int64,copy=True)).to(device)
        dow=torch.from_numpy(np.array(ds.dow[ids],dtype=np.int64,copy=True)).to(device)
        pred,_=teacher(x*m,m,tod,dow)
        target=np.nan_to_num(np.asarray(ds.y_norm[ids],np.float32))
        forecast=pred.float().cpu().numpy()
        if not np.isfinite(forecast).all():
            raise RuntimeError(f'Non-finite STLA normal teacher predictions at rows {s}:{e}')
        y[s:e]=np.where(np.asarray(ds.valid[ids],bool),target-forecast,0.).astype('float32')
        if e==len(ds) or (s//chunk)%10==0:_checkpoint_arrays(arrays,progress,e)
    _checkpoint_arrays(arrays,progress,len(ds));del arrays;del y
    os.replace(Path(out)/'normal.partial.npy',p)
    progress.unlink(missing_ok=True)
    return p

@torch.inference_mode()
def prepare_view(cache,split,view,teacher,device,root,shortlist,k=12,chunk=8,
                 replay_chunk=48,seed=10,limit=None,regions=None,teacher_sha='',
                 history_cache=None,split_key=None):
    ds=SupportSplit(cache,split,limit,cf_training=(split=='train' and view>0))
    if split!='train' and view!=0:raise ValueError('Paired views are train-only')
    target=_paths(root,split_key or split,view);target.mkdir(parents=True,exist_ok=True)
    done=target/'manifest.json'
    from ..config import file_sha256
    shortlist_sha=('visible_retrieval_v92_'+file_sha256(Path(__file__).with_name('retrieval.py'))
                    if shortlist is None or view>0 else file_sha256(shortlist)+'_shared_anchor_v92')
    history_root=str(Path(history_cache or cache).resolve())
    mask_protocol_sha=file_sha256(Path(__file__).parents[1]/'v9'/'views.py')
    region_fingerprint=hashlib.sha256(np.ascontiguousarray(regions).tobytes()).hexdigest() if regions is not None else None
    if done.exists():
        old=json.loads(done.read_text())
        if (old.get('teacher_sha')!=teacher_sha or old.get('rows')!=len(ds) or
            old.get('k')!=k or old.get('seed')!=seed or
            old.get('shortlist_sha')!=shortlist_sha or old.get('history_cache')!=history_root or
            old.get('mask_protocol_sha')!=mask_protocol_sha or old.get('region_fingerprint')!=region_fingerprint):
            raise RuntimeError(f'Stale v9 view cache: {target}. Remove exact version folder to rebuild.')
        return PreparedView(target)
    normal_rows=np.load(root/'normal_train.npy',mmap_mode='r').shape[0]
    history_cache=history_cache or cache
    mem=FutureMemory(history_cache,limit=normal_rows)
    if k>len(mem):raise ValueError('K exceeds history memory')
    hh=ds.x_obs.shape[1];n=ds.x_obs.shape[-1];h=ds.y_norm.shape[1]
    if split=='train' and view>0 and ds.x_full is None:raise ValueError('Missing full training data')
    retriever=VisibleCandidateRetriever(history_cache,limit=normal_rows,shared=min(4,k))
    official_retrieval=RetrievalCache(shortlist) if view==0 and shortlist is not None else None
    if official_retrieval is not None and official_retrieval.candidate_count!=k:
        raise ValueError('Official strong retrieval must contain K candidates')
    specs={'base':('float32',(len(ds),h,n)),
           'context':('float32',(len(ds),n,teacher.model_dim)),
           'cf':('float32',(len(ds),k,h,n)),
           'candidate_ids':('int32',(len(ds),k)),
           'scores':('float32',(len(ds),k)),
           'valid':('bool',(len(ds),k)),
           'match':('float32',(len(ds),k,n,2)),
           'mask':('float32',(len(ds),hh,n)),
           'x':('float32',(len(ds),hh,n)),
           'teacher_cf':('float32',(len(ds),h,n)),
           'family':('int16',(len(ds),)),
           'severity':('int16',(len(ds),)),
           'mask_hash':('S64',(len(ds),))}
    arrays,first_row,progress=_resume_memmaps(target,specs)
    starts=ds.window_starts
    train_starts=np.array(SupportSplit(history_cache,'train').window_starts[:normal_rows],dtype=np.int64)
    t0=time.perf_counter()
    bank=np.load(root/'normal_train.npy',mmap_mode='r')
    # Every view chooses its own shortlist using its own visible query.
    # Four shared anchors are chosen using history timestamps only.
    for s in range(first_row,len(ds),chunk):
        e=min(s+chunk,len(ds));idx=np.arange(s,e);bb=len(idx)
        original=source_mask(ds,idx)
        family_ids=np.zeros(bb,dtype=np.int16)
        severity_values=np.zeros(bb,dtype=np.int16)
        if view==0:
            qmask=np.array(ds.observed[idx],dtype=np.float32,copy=True)
            x=np.nan_to_num(np.array(ds.x_obs[idx],dtype=np.float32,copy=True))
            cand=np.empty((bb,k),np.int32);scores=np.empty((bb,k),np.float32)
            for a in range(bb):
                if official_retrieval is None:
                    cand[a],scores[a]=retriever.select(x[a],qmask[a],starts[idx[a]],k)
                else:
                    # Retain up to K-4 candidates from the strong original v5
                    # masked-query retrieval, plus four independently time-legal
                    # anchors guaranteed to appear in BOTH forecasting views.
                    anchor,anchor_scores=retriever.select(x[a],qmask[a],starts[idx[a]],min(4,k))
                    chosen=list(map(int,anchor));lookup=list(map(float,anchor_scores));used=set(chosen)
                    candidates=official_retrieval.selected_indices[int(idx[a])]
                    cscores=official_retrieval.selected_scores[int(idx[a])]
                    for cid,sc in zip(candidates,cscores):
                        if len(chosen)>=k:break
                        if int(cid) not in used:
                            chosen.append(int(cid));lookup.append(float(sc));used.add(int(cid))
                    if len(chosen)<k:
                        extra,extra_scores=retriever.select(x[a],qmask[a],starts[idx[a]],k)
                        for cid,sc in zip(extra,extra_scores):
                            if len(chosen)>=k:break
                            if int(cid) not in used:
                                chosen.append(int(cid));lookup.append(float(sc));used.add(int(cid))
                    if len(chosen)<k:
                        chosen.extend([0]*(k-len(chosen)));lookup.extend([-1e4]*(k-len(lookup)))
                    cand[a]=chosen;scores[a]=lookup
        else:
            qmask=np.zeros((bb,hh,n),dtype='float32')
            for a,qi in enumerate(idx):
                m,family,severity=paired_mask(original[a],int(qi),seed,view,region=regions)
                qmask[a]=m.astype('float32')
                family_ids[a]=1+FAMILIES.index(family)
                severity_values[a]=severity
            x=np.nan_to_num(np.array(ds.x_full[idx],dtype=np.float32,copy=True))*qmask
            cand=np.empty((bb,k),np.int32);scores=np.empty((bb,k),np.float32)
            for a in range(bb):cand[a],scores[a]=retriever.select(x[a],qmask[a],starts[idx[a]],k)
        # Validate every source candidate's future matures before the query
        in_range=(cand>=0)&(cand<len(train_starts))
        safe_for_time=np.where(in_range,cand,0)
        legal_end=train_starts[safe_for_time].astype('int64')+hh+h
        valid_ids=(legal_end<=starts[idx,None])&in_range
        # Candidate duplication can occur early in a chronological series
        # when fewer than K mature examples exist. Keep first, mask repeats.
        for row in range(bb):
            seen=set()
            for col in range(k):
                cid=int(cand[row,col])
                if cid in seen:valid_ids[row,col]=False
                else:seen.add(cid)
        # v5 shortlist can contain future/overlapping TRAIN candidates. They are
        # explicitly suppressed in v9 instead of being allowed to leak.
        if not valid_ids.all():
            scores=np.where(valid_ids,scores,-1e4).astype('float32')
        x_t=torch.from_numpy(x).to(device);m_t=torch.from_numpy(qmask).to(device)
        tod=torch.from_numpy(np.array(ds.tod[idx],np.int64,copy=True)).to(device)
        dow=torch.from_numpy(np.array(ds.dow[idx],np.int64,copy=True)).to(device)
        bp,ctx=teacher(x_t,m_t,tod,dow)
        if not (torch.isfinite(bp).all() and torch.isfinite(ctx).all()):
            raise RuntimeError(f'Non-finite STLA view predictions/context at {split}/view{view} rows {s}:{e}')
        safe_cand=cand.copy()
        safe_cand[~valid_ids]=0  # never read a future/overlapping historical label
        candidate=mem.fetch(torch.from_numpy(safe_cand.astype(np.int64)),device)
        complete_future=candidate['future_valid'].bool().flatten(2).all(-1).cpu().numpy()
        valid_ids &= complete_future
        safe_cand[~valid_ids]=0
        scores=np.nan_to_num(np.where(valid_ids,scores,-1e4).astype('float32'),
                nan=-1e4,posinf=1e4,neginf=-1e4)
        replay=counterfactual_residuals(teacher,candidate,m_t,chunk_size=replay_chunk,
                      replay='matched',strict_source_mask=True)
        # historical normal residual from train bank, loaded only for this chunk
        norm=torch.from_numpy(np.array(bank[safe_cand],np.float32,copy=True)).to(device)
        cf=(replay.float()-norm)*torch.from_numpy(valid_ids).to(device)[:,:,None,None]
        if not torch.isfinite(cf).all():
            raise RuntimeError(f'Non-finite historical CF replay at {split}/view{view} rows {s}:{e}')
        srcmask=candidate['source_valid'].bool()
        common=m_t[:,None].bool()&srcmask
        diff=(x_t[:,None]-candidate['x']).abs()
        match_sum=(diff*common).sum(2)
        denom=common.float().sum(2)
        agreement=torch.exp(-match_sum/denom.clamp_min(1))
        has=(denom>0).float()
        match=torch.stack([agreement*has,has],-1)
        match=match*torch.from_numpy(valid_ids).to(device)[:,:,None,None]
        # TRAIN-only full teacher target, never fed to TCRRV9.forward().
        if split=='train':
            normal_query=np.array(bank[idx],np.float32,copy=True)
            valid_target=np.asarray(ds.valid[idx],bool)
            normal_bp=np.nan_to_num(np.array(ds.y_norm[idx],np.float32,copy=True))-normal_query
            query_cf=np.where(valid_target,normal_bp-bp.float().cpu().numpy(),0.).astype('float32')
        else:
            query_cf=np.zeros((bb,h,n),dtype='float32')
        # If no historical overlap, no agreement evidence; graph context remains available.
        hashes=np.array([mask_sha(qmask[row]).encode('ascii') for row in range(bb)],dtype='S64')
        data=dict(family=family_ids,severity=severity_values,mask_hash=hashes,
                  teacher_cf=query_cf,base=bp.float().cpu().numpy(),context=ctx.float().cpu().numpy(),
                  cf=cf.cpu().numpy(),candidate_ids=safe_cand,scores=scores,
                  valid=valid_ids,match=match.cpu().numpy(),mask=qmask,x=x)
        for name,value in data.items():arrays[name][s:e]=value
        if e==len(ds) or (s//chunk)%25==0:
            _checkpoint_arrays(arrays,progress,e)
            print(f'V9_PREP split={split} view={view} {e}/{len(ds)} '
                  f'elapsed={time.perf_counter()-t0:.1f}s',flush=True)
    _checkpoint_arrays(arrays,progress,len(ds))
    del arrays
    for name in specs:os.replace(target/(name+'.partial.npy'),target/(name+'.npy'))
    progress.unlink(missing_ok=True)
    meta=dict(schema=PREPARATION_SCHEMA,version='9.2.0',split=split,view=view,rows=len(ds),k=k,h=h,n=n,
              teacher_sha=teacher_sha,seed=seed,physical_regions=regions is not None,
              history_cache=str(Path(history_cache).resolve()),shortlist=str(shortlist),
              shortlist_sha=shortlist_sha,mask_protocol_sha=mask_protocol_sha,
              region_fingerprint=region_fingerprint,paired_candidate_protocol='strong_official_plus_view_specific_pair_with_shared_legal_anchors_v92',
              failure_names=['official']+list(FAMILIES),
              family_counts={str(i):int(np.count_nonzero(np.load(target/'family.npy',mmap_mode='r')==i))
                             for i in range(len(FAMILIES)+1)},
              valid_candidates=int(np.load(target/'valid.npy',mmap_mode='r').sum()))
    tmp=target/'manifest.json.tmp';tmp.write_text(json.dumps(meta,indent=2));os.replace(tmp,done)
    return PreparedView(target)

class V9PairedDataset(torch.utils.data.Dataset):
    """Same query ID, two distinct sensing failures; candidate IDs remain explicit."""
    def __init__(self,cache,root,limit=None):
        self.data=V9EvidenceDataset(cache,'train',(0,1),limit=limit,root=root)
    def __len__(self):return len(self.data)//2
    def __getitem__(self,i):return {'official':self.data[2*i], 'paired':self.data[2*i+1]}
