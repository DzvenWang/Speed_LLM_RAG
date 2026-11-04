"""Multi-resolution TRAIN-memory retrieval and query-conditioned counterfactual replay.

No query future read in candidate selection or counterfactual construction.
Every chunk has hashed artifacts and a fail-closed identity manifest.
"""
from __future__ import annotations
import os,json,hashlib,functools
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from ..data import SupportSplit
from ..config import cache_path,file_sha256
from .storage import file_manifest
from .support import pair_support_features

KEYS=('normal','cf','scores','coverage','ids','valid','support_raw')

def signature(x,mask,period=3,bins=12):
    """[B,C] history/mask statistics; period in 1,2,4 (or other divisors)."""
    if x.shape!=mask.shape:raise ValueError('History mask mismatch')
    B,H,N=x.shape;vals=[]
    x=torch.nan_to_num(x.float())*mask.float();mask=mask.float()
    for start in range(0,H,period):
        xx=x[:,start:start+period];mm=mask[:,start:start+period]
        for i in range(min(bins,N)):
            l=i*N//min(bins,N);r=(i+1)*N//min(bins,N)
            m=mm[:,:,l:r]
            vals.extend((xx[:,:,l:r].sum((1,2))/m.sum((1,2)).clamp_min(1),m.mean((1,2))))
    return torch.stack(vals,-1)

def _gather(a,idx,device,dtype=None):
    ii=idx.detach().cpu().numpy().ravel()
    raw=np.asarray(a[ii],dtype=dtype).copy()
    return torch.from_numpy(raw).to(device).reshape(*idx.shape,*raw.shape[1:])

def _write_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.partial');t.write_text(json.dumps(obj,indent=2,sort_keys=True,default=str))
    os.replace(t,p)

def _hash_obj(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()

def _candidate_select(query_x,query_mask,query_start,train_keys,train_start,k,shortlist,
                      periods,source_x,source_valid,source_y,source_future_valid,
                      forecast,history_weight=.6,forecast_weight=.2):
    """Legal candidate shortlist and reranking. B small, TRAIN memories on GPU.

    Historical future is from TRAIN and may be used for ranking; `forecast`
    must be computed with query observed history only.
    """
    device=query_x.device;B,H,N=query_x.shape
    prohibit=((train_start[None]<query_start[:,None]+H+forecast.shape[1])&
              (query_start[:,None]<train_start[None]+H+forecast.shape[1]))
    ids_bank=[];score_bank=[]
    for period,keys in zip(periods,train_keys):
        qkey=signature(query_x,query_mask,period=period)
        dist=torch.cdist(qkey.float(),keys.float())/(keys.shape[-1]**.5)
        dist.masked_fill_(prohibit,float('inf'))
        vals,inds=torch.topk(dist,min(shortlist,keys.shape[0]),dim=1,largest=False)
        ids_bank.append(inds);score_bank.append(-vals)
    ids=torch.cat(ids_bank,-1);hs=torch.cat(score_bank,-1)
    sx=torch.nan_to_num(_gather(source_x,ids,device,np.float32))
    sv=_gather(source_valid,ids,device,np.float32).float()
    overlap=sv*query_mask[:,None]
    cmp=((sx-query_x[:,None]).abs()*overlap).sum((2,3))/overlap.sum((2,3)).clamp_min(1)
    matched=torch.exp(-cmp.clamp(0,15))
    matched=torch.where(overlap.sum((2,3))>0,matched,matched.new_zeros(matched.shape))
    sy=torch.nan_to_num(_gather(source_y,ids,device,np.float32))
    fy=_gather(source_future_valid,ids,device,np.float32)
    q=forecast[:,None].float()
    # Candidate future shape must remain based only on TRAIN labels and valid mask.
    y=sy.float();v=fy.float()
    y_mean=(y*v).sum((2,3),keepdim=True)/v.sum((2,3),keepdim=True).clamp_min(1)
    q_mean=(q*v).sum((2,3),keepdim=True)/v.sum((2,3),keepdim=True).clamp_min(1)
    yc=(y-y_mean)*v;qc=(q-q_mean)*v
    similarity=(yc*qc).sum((2,3))/(yc.square().sum((2,3)).sqrt()*qc.square().sum((2,3)).sqrt()+1e-6)
    similarity=torch.nan_to_num(similarity,nan=0).clamp(-1,1)
    # Lower observation coverage reduces reliance on an uncertain query forecast.
    fw=forecast_weight*query_mask.mean((1,2)).sqrt()[:,None]
    score=history_weight*hs+(1-history_weight)*matched+fw*similarity
    if (~torch.isfinite(score)).all(-1).any():raise RuntimeError('No finite candidate scores')
    # Deduplicate top candidates; rerank is deterministic for ties.
    selected=[];selected_scores=[]
    ids_np=ids.cpu().numpy();score_np=score.cpu().numpy()
    for b in range(B):
        order=np.argsort(-score_np[b],kind='stable');seen=set();row=[]
        for j in order:
            idx=int(ids_np[b,j])
            if not np.isfinite(score_np[b,j]) or idx in seen:continue
            seen.add(idx);row.append(int(j))
            if len(row)==k:break
        if len(row)!=k:raise RuntimeError('Not enough nonoverlapping unique TRAIN candidates')
        selected.append(ids_np[b,row]);selected_scores.append(score_np[b,row])
    return (torch.tensor(np.asarray(selected,dtype=np.int64),device=device),
            torch.tensor(np.asarray(selected_scores,dtype=np.float32),device=device))

@torch.inference_mode()
def prepare_replay(args,device,split,severity=None):
    from .runner import train_cache,make_teacher,graph_and_memory
    name='test' if severity is not None else split
    cache=cache_path(args.config,'evaluation',severity) if severity is not None else cache_path(args.config,'formal')
    query=SupportSplit(cache,name,limit=args.limit('test' if severity is not None else split))
    train,train_records,_=train_cache(args,device,'train')
    if train.x_full is None:raise RuntimeError('TRAIN complete histories required for historical memory')
    if train.original_valid is None:raise RuntimeError('TRAIN original history-valid mask required')
    _,graph,_=graph_and_memory(args,device)
    full=np.load(train.root/'train_x_full_norm.npy',mmap_mode='r')
    source_valid=np.asarray(train.observed[:len(train)],bool) | np.asarray(train.original_valid[:len(train)],bool)
    # Build TRAIN signatures once per period, streaming to prevent GPU spikes.
    periods=tuple(args.retrieval_periods)
    config={
        'schema':'v7_replay_v1','implementation_sha256':file_sha256(Path(__file__)),
        'source':file_manifest(query,name,file_sha256(args.run/'s0.pt'),2),
        'train_source':file_manifest(train,'train',file_sha256(args.run/'s0.pt'),2),
        'graph_sha256':file_sha256(args.run/'train_only_neighbours.pt'),
        's0_sha256':file_sha256(args.run/'s0.pt'),
        'checkpoint_sha256':file_sha256(args.backbone_checkpoint),
        'k':args.k,'periods':list(periods),'shortlist':args.retrieval_shortlist,
        'history_weight':args.retrieval_history_weight,
        'forecast_weight':args.retrieval_forecast_weight,
        'chunk_queries':args.replay_chunk,'rows':len(query),'train_rows':len(train),
        'dataset':args.dataset,'target':args.target,'severity':severity,
    }
    root=args.run/'paired_replay'/('test_s'+str(severity) if severity is not None else split)
    root.mkdir(parents=True,exist_ok=True)
    identity=_hash_obj(config);meta=root/'identity.json'
    if meta.exists() and json.loads(meta.read_text())!=config:
        raise RuntimeError('Replay identity mismatch; choose fresh run directory')
    if not meta.exists():_write_json(meta,config)
    valid_tensor=torch.from_numpy(source_valid)
    keys=[]
    keyfiles=args.run/'retrieval_keys';keyfiles.mkdir(exist_ok=True)
    for period in periods:
        path=keyfiles/f'signature_p{period}.pt';identity_path=keyfiles/f'signature_p{period}.json'
        key_ident={'train_source':config['train_source'],'period':period,'implementation_sha256':config['implementation_sha256']}
        if path.exists() and identity_path.exists() and json.loads(identity_path.read_text())==key_ident:
            keys.append(torch.load(path,map_location='cpu',weights_only=True).to(device));continue
        parts=[]
        for a in range(0,len(train),128):
            b=min(a+128,len(train))
            xs=torch.from_numpy(np.asarray(full[a:b],np.float32).copy()).to(device)
            ms=valid_tensor[a:b].to(device).float()
            parts.append(signature(xs,ms,period=period).cpu())
        entry=torch.cat(parts,0)
        tmp=path.with_suffix('.partial.pt');torch.save(entry,tmp);os.replace(tmp,path)
        _write_json(identity_path,key_ident);keys.append(entry.to(device))
    starts=torch.from_numpy(np.array(train.window_starts[:len(train)],copy=True)).to(device)
    neighbor=graph['functional'].to(device)
    neighbor_valid=graph['functional_valid'].to(device)
    teacher=None
    for start in range(0,len(query),args.replay_chunk):
        stop=min(start+args.replay_chunk,len(query));d=root/f'chunk_{start:06d}_{stop:06d}';done=d/'DONE.json'
        good=False
        if done.exists():
            try:
                info=json.loads(done.read_text());good=info.get('identity')==identity
                for key in KEYS:
                    f=d/(key+'.npy');a=np.load(f,mmap_mode='r')
                    shp=((stop-start,args.k,args.horizon,args.nodes) if key in ('normal','cf','valid') else
                         (stop-start,args.k,args.nodes,14) if key=='support_raw' else
                         (stop-start,args.k))
                    if a.shape!=shp or file_sha256(f)!=info.get('hashes',{}).get(key):good=False
            except (OSError,ValueError,KeyError):good=False
        if good:
            print(f'V7_REPLAY_REUSE {name} {stop}/{len(query)}',flush=True);continue
        if teacher is None:
            teacher_state=torch.load(args.run/'s0.pt',map_location='cpu',weights_only=False)
            teacher,_=make_teacher(args,device,teacher_state);teacher.eval()
        d.mkdir(parents=True,exist_ok=True);done.unlink(missing_ok=True)
        qx=torch.from_numpy(np.asarray(query.x_obs[start:stop],np.float32).copy()).to(device)
        qm=torch.from_numpy(np.asarray(query.observed[start:stop],np.float32).copy()).to(device)
        qt=torch.from_numpy(np.asarray(query.tod[start:stop]).copy()).to(device).long()
        qd=torch.from_numpy(np.asarray(query.dow[start:stop]).copy()).to(device).long()
        qstart=torch.from_numpy(np.asarray(query.window_starts[start:stop]).copy()).to(device)
        # target-only forecast is legal, obtained before retrieval; never uses y/query future.
        forecasts=[]
        for i in range(0,len(qx),args.replay_batch):
            out,_=teacher(qx[i:i+args.replay_batch],qm[i:i+args.replay_batch],
                          qt[i:i+args.replay_batch],qd[i:i+args.replay_batch])
            forecasts.append(out.float())
        qpred=torch.cat(forecasts,0)
        ids,scores=_candidate_select(qx,qm,qstart,keys,starts,args.k,args.retrieval_shortlist,
                   periods,full,source_valid,train.y_norm,train.valid,qpred,
                   args.retrieval_history_weight,args.retrieval_forecast_weight)
        sx=torch.nan_to_num(_gather(full,ids,device,np.float32))
        sv=_gather(source_valid,ids,device,np.float32).bool()
        raw=pair_support_features(qx,qm,sx,sv,neighbor,neighbor_valid)
        src_tod=_gather(train.tod,ids,device).long().reshape(-1,args.history)
        src_dow=_gather(train.dow,ids,device).long().reshape(-1,args.history)
        source_normal=_gather(train_records['normal'],ids,device,np.float32)
        hist_y=torch.nan_to_num(_gather(train.y_norm,ids,device,np.float32))
        reference=torch.nan_to_num(hist_y-source_normal)
        valid=_gather(train.valid,ids,device,np.bool_)
        paired_mask=sv.float()*qm[:,None]
        xc=(sx*paired_mask).reshape(-1,args.history,args.nodes)
        mc=paired_mask.reshape(-1,args.history,args.nodes)
        pred=[]
        for i in range(0,len(xc),args.replay_batch):
            y,_=teacher(xc[i:i+args.replay_batch],mc[i:i+args.replay_batch],
                        src_tod[i:i+args.replay_batch],src_dow[i:i+args.replay_batch])
            pred.append(y.float())
        counterfactual=reference-torch.cat(pred).reshape(stop-start,args.k,args.horizon,args.nodes)
        arrays={'normal':source_normal,'cf':counterfactual,
                'scores':scores,'coverage':sv.float().mean((2,3)),
                'ids':ids.int(),'valid':valid,'support_raw':raw}
        for key,x in arrays.items():
            if not torch.isfinite(x.float()).all():raise RuntimeError(f'Nonfinite replay {key}')
            dtype=(np.float16 if key in ('normal','cf','support_raw') else
                   np.float32 if key in ('scores','coverage') else np.int32 if key=='ids' else np.bool_)
            dest=d/(key+'.npy');tmp=d/(key+'.partial.npy')
            np.save(tmp,x.detach().cpu().numpy().astype(dtype,copy=False),allow_pickle=False)
            os.replace(tmp,dest)
        _write_json(done,{'identity':identity,'start':start,'end':stop,
                         'hashes':{key:file_sha256(d/(key+'.npy')) for key in KEYS}})
        print(f'V7_REPLAY_PROGRESS split={name} severity={severity} {stop}/{len(query)}',flush=True)
    if teacher is not None:
        del teacher
        if device.type=='cuda':torch.cuda.empty_cache()
    return root

class ReplayDataset(Dataset):
    def __init__(self,source,records,root,chunk_size):
        self.source=source;self.records=records;self.root=Path(root);self.chunk=int(chunk_size)
        conf=json.loads((self.root/'identity.json').read_text())
        if len(source)!=conf['rows'] or self.chunk!=conf['chunk_queries']:
            raise RuntimeError('Replay Dataset identity/shape mismatch')
        self.length=len(source)
        for i in range(0,self.length,self.chunk):
            if not (self.root/f'chunk_{i:06d}_{min(self.length,i+self.chunk):06d}'/'DONE.json').exists():
                raise RuntimeError('Incomplete replay; run replay stage')
    def __len__(self):return self.length
    @functools.lru_cache(maxsize=12)
    def _arrays(self,chunk):
        a=chunk*self.chunk;b=min(a+self.chunk,self.length)
        path=self.root/f'chunk_{a:06d}_{b:06d}'
        return {k:np.load(path/(k+'.npy'),mmap_mode='r') for k in KEYS}
    def __getitem__(self,i):
        row=self.source[i]
        for key,v in self.records.items():
            row['cached_'+key]=torch.from_numpy(np.asarray(v[i],np.float32).copy())
        row.pop('x_full_training_only',None)
        arrays=self._arrays(i//self.chunk)
        for key in KEYS:
            dtype=np.bool_ if key=='valid' else np.int64 if key=='ids' else np.float32
            row['candidate_'+key]=torch.from_numpy(np.asarray(arrays[key][i%self.chunk],dtype=dtype).copy())
        return row
