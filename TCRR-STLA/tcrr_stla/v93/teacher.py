"""Full-reference multi-mask teacher caches, with chunk-level restart and no val full data.

All teacher targets are TRAIN ONLY. Validation stores official masked base/context only.
"""
from __future__ import annotations
import os,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from ..data import SupportSplit
from ..config import file_sha256
from .interventions import FAMILIES,SEVERITIES,observed_intervention,plan_digest

SCHEMA=930


def atomic_json(path,obj):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,sort_keys=True,indent=2,default=str));os.replace(tmp,p)


def source_valid(ds,ids):
    observed=np.asarray(ds.observed[ids],dtype=bool)
    artificial=np.asarray(ds.original_valid[ids],dtype=bool) if ds.original_valid is not None else np.zeros_like(observed)
    return observed|artificial


def build_identity(cache,project_root,teacher_path,adapter_path,target,dataset,seed,views,limit,val_limit=None):
    names=('train_x_full_norm.npy','train_x_obs_norm.npy','train_history_observed.npy',
           'train_artificial_missing.npy','train_y_norm.npy','train_future_valid.npy',
           'train_tod_hist.npy','train_dow_hist.npy','train_direct_mask.npy','train_manifest.csv',
           'val_x_obs_norm.npy','val_y_norm.npy','val_future_valid.npy','val_direct_mask.npy',
           'val_history_observed.npy','val_tod_hist.npy',
           'val_dow_hist.npy','val_manifest.csv')
    hashes={name:file_sha256(Path(cache)/name) for name in names if (Path(cache)/name).is_file()}
    essential=set(names)-{'train_artificial_missing.npy'}
    if not essential.issubset(hashes):raise RuntimeError(f'Missing experiment cache files: {sorted(essential-set(hashes))}')
    return dict(schema=SCHEMA,target=target,dataset=dataset,seed=int(seed),views=int(views),
                limit=int(limit) if limit else None,val_limit=int(val_limit) if val_limit else None,
                cache=str(Path(cache).resolve()),data=hashes,
                teacher=file_sha256(teacher_path),adapter=file_sha256(adapter_path),
                intervention_plan=plan_digest(project_root,target,dataset,views))


def memmaps_for(folder,specs,identity):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    done=folder/'manifest.json';state=folder/'progress.json';ident=folder/'identity.json'
    if done.exists():
        if json.loads(ident.read_text())!=identity:raise RuntimeError('Completed teacher cache identity differs')
        meta=json.loads(done.read_text())
        if meta['schema']!=SCHEMA:raise RuntimeError('Teacher cache schema mismatch')
        for key,(dtype,shape) in specs.items():
            a=np.load(folder/(key+'.npy'),mmap_mode='r',allow_pickle=False)
            if a.shape!=shape or a.dtype!=np.dtype(dtype):raise RuntimeError(f'Broken completed cache {key}')
        return None,next(iter(specs.values()))[1][0],True
    if ident.exists() and json.loads(ident.read_text())!=identity:
        raise RuntimeError('Partial teacher cache identity mismatch; use a new folder')
    if not ident.exists():
        if state.exists() or any(folder.glob('*.partial.npy')):raise RuntimeError('Orphan teacher cache partial files')
        atomic_json(ident,identity)
    if state.exists():
        progress=json.loads(state.read_text())
        if progress['shapes']!={key:list(shape) for key,(_,shape) in specs.items()}:
            raise RuntimeError('Partial teacher shapes mismatch')
        maps={k:np.lib.format.open_memmap(folder/(k+'.partial.npy'),mode='r+') for k in specs}
        for k,(dtype,shape) in specs.items():
            if maps[k].shape!=shape or maps[k].dtype!=np.dtype(dtype):raise RuntimeError('Partial teacher dtype/shape mismatch')
        return maps,int(progress['next_row']),False
    maps={k:np.lib.format.open_memmap(folder/(k+'.partial.npy'),mode='w+',dtype=dtype,shape=shape)
          for k,(dtype,shape) in specs.items()}
    atomic_json(state,{'next_row':0,'shapes':{k:list(shape) for k,(_,shape) in specs.items()}})
    return maps,0,False


def flush_rows(folder,maps,next_row):
    for a in maps.values():a.flush()
    state=Path(folder)/'progress.json';obj=json.loads(state.read_text());obj['next_row']=int(next_row);atomic_json(state,obj)


def complete_maps(folder,maps,meta):
    folder=Path(folder)
    flush_rows(folder,maps,meta['rows'])
    del maps
    for p in folder.glob('*.partial.npy'):
        os.replace(p,folder/p.name.replace('.partial.npy','.npy'))
    atomic_json(folder/'manifest.json',meta)
    (folder/'progress.json').unlink(missing_ok=True)


def _forward(teacher,x,m,tod,dow):
    y,z=teacher(x*m,m,tod,dow)
    if not torch.isfinite(y).all() or not torch.isfinite(z).all():
        raise RuntimeError('Teacher output contains nonfinite values')
    return y.float(),z.float()


@torch.inference_mode()
def prepare_split(cache,folder,split,teacher,device,project_root,target,dataset,seed,
                  views=4,batch_size=4,limit=None,identity=None):
    if split not in ('train','val'):raise ValueError('Prepare train or val only')
    if split=='val' and views!=1:raise ValueError('Validation cannot have synthetic intervention views')
    ds=SupportSplit(cache,split,limit,cf_training=(split=='train'))
    nrows=len(ds);L,N=ds.observed.shape[1:];H=ds.y_norm.shape[1]
    teacher.eval()
    if nrows==0:raise RuntimeError('Empty teacher dataset')
    # Use one masked official view to infer the real STLA context dimension.
    x0=torch.as_tensor(np.nan_to_num(np.asarray(ds.x_obs[:1],dtype=np.float32)),device=device)
    m0=torch.as_tensor(np.asarray(ds.observed[:1],dtype=np.float32),device=device)
    tod0=torch.as_tensor(np.array(ds.tod[:1],dtype=np.int64,copy=True),device=device)
    dow0=torch.as_tensor(np.array(ds.dow[:1],dtype=np.int64,copy=True),device=device)
    forecast,context=_forward(teacher,x0,m0,tod0,dow0)
    C=int(context.shape[-1]);assert forecast.shape==(1,H,N) and context.shape==(1,N,C)
    specs={'base':('float32',(nrows,views,H,N)),
           'context':('float16',(nrows,views,N,C)),
           'visible':('uint8',(nrows,views,L,N)),
           'family':('int8',(nrows,views)),
           'severity':('int16',(nrows,views))}
    if split=='train':specs.update({'reference_base':('float32',(nrows,H,N)),
                                    'reference_residual':('float32',(nrows,H,N)),
                                    'cf_response':('float32',(nrows,views,H,N)),
                                    'error_damage':('float32',(nrows,views,H,N))})
    folder=Path(folder);maps,start,already=memmaps_for(folder,specs,identity)
    if already:
        print(f'V93_TEACHER_REUSE split={split} rows={nrows}',flush=True);return
    for s in range(start,nrows,batch_size):
        e=min(s+batch_size,nrows);ids=np.arange(s,e)
        tod=torch.as_tensor(np.array(ds.tod[ids],dtype=np.int64),device=device)
        dow=torch.as_tensor(np.array(ds.dow[ids],dtype=np.int64),device=device)
        if split=='train':
            reference_valid=source_valid(ds,ids)
            x_raw=np.array(ds.x_full[ids],dtype=np.float32,copy=True)
            if not np.isfinite(x_raw[reference_valid]).all():
                raise RuntimeError('Nonfinite source measurements flagged valid in TRAIN reference')
            x_raw=np.where(reference_valid,np.nan_to_num(x_raw,nan=0,posinf=0,neginf=0),0.)
            # The official masked observed values must match the reference where visible.
            official=np.asarray(ds.observed[ids],dtype=bool)
            if np.any(official & ~reference_valid):raise RuntimeError('Official mask not within reference')
            official_raw=np.array(ds.x_obs[ids],dtype=np.float32,copy=True)
            if not np.allclose(x_raw[official],official_raw[official],rtol=1e-4,atol=1e-4):
                raise RuntimeError('Full history and official masked input disagree on visible measurements')
            ref=torch.as_tensor(reference_valid.astype('float32'),device=device)
            x=torch.as_tensor(x_raw,device=device)
            bref,_=_forward(teacher,x,ref,tod,dow)
            b_ref=bref.cpu().numpy()
            future_valid=np.asarray(ds.valid[ids],dtype=bool)
            yn=np.array(ds.y_norm[ids],dtype=np.float32,copy=True)
            if not np.isfinite(yn[future_valid]).all():raise RuntimeError('Invalid finite TRAIN targets')
            ref_residual=yn-b_ref
            maps['reference_base'][s:e]=b_ref
            maps['reference_residual'][s:e]=np.where(future_valid,ref_residual,0.)
        else:
            xraw=np.array(ds.x_obs[ids],dtype=np.float32,copy=True)
            x=torch.as_tensor(np.nan_to_num(xraw,nan=0,posinf=0,neginf=0),device=device)
            reference_valid=np.asarray(ds.observed[ids],dtype=bool)
            future_valid=np.asarray(ds.valid[ids],dtype=bool)
        # Avoid full [B*V,L,N] feature materialization on V100.
        for j in range(views):
            if j==0:
                visible=np.asarray(ds.observed[ids],dtype=bool)
                family=0;severity=0
            else:
                masks=[];family_ids=[];severities=[]
                for r,qi in enumerate(ids):
                    mm,spec=observed_intervention(reference_valid[r],int(qi),int(ds.window_starts[qi]),
                        j,views,dataset,project_root,target)
                    masks.append(mm);family_ids.append(1+FAMILIES.index(spec['family']))
                    severities.append(int(round(100*float(spec['severity']))))
                visible=np.stack(masks);family=np.asarray(family_ids,dtype=np.int8)
                severity=np.asarray(severities,dtype=np.int16)
            m=torch.as_tensor(visible.astype('float32'),device=device)
            b,z=_forward(teacher,x,m,tod,dow)
            bj=b.cpu().numpy()
            maps['base'][s:e,j]=bj
            maps['context'][s:e,j]=z.cpu().numpy().astype(np.float16)
            maps['visible'][s:e,j]=visible.astype(np.uint8)
            maps['family'][s:e,j]=family
            maps['severity'][s:e,j]=severity
            if split=='train':
                cf=b_ref-bj
                masked_residual=yn-bj
                deviation=ref_residual+cf-masked_residual
                if np.max(np.abs(deviation[future_valid]),initial=0.)>1e-4:
                    raise RuntimeError('Full-to-masked counterfactual identity failed')
                maps['cf_response'][s:e,j]=np.where(future_valid,cf,0.)
                damage=np.abs(masked_residual)-np.abs(ref_residual)
                maps['error_damage'][s:e,j]=np.where(future_valid,damage,0.)
        flush_rows(folder,maps,e)
        if e==nrows or (s//batch_size)%25==0:
            print(f'V93_TEACHER_PROGRESS split={split} rows={e}/{nrows} views={views}',flush=True)
    complete_maps(folder,maps,{'schema':SCHEMA,'split':split,'rows':nrows,'views':views,
                    'history':L,'horizon':H,'nodes':N,'context_dim':C,'source':'frozen_STLA_reference_multimask'})
    print(f'V93_TEACHER_PASS split={split} rows={nrows} views={views}',flush=True)
