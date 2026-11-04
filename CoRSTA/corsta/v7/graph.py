"""TCRR-owned, read-only-source, deterministic spatial-graph builder.

Does not reuse Base_SupportBench directed_neighbors.npz: its missing-distance
fallback is on a different scale from 1/d. Never changes input data or masks.
"""
from __future__ import annotations
from pathlib import Path
import hashlib, json, os
import numpy as np
import pandas as pd
import torch

GRAPH_SCHEMA=1

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def haversine_m(lat,lon):
    a=np.deg2rad(lat.astype(np.float64)); b=np.deg2rad(lon.astype(np.float64))
    da=a[:,None]-a[None,:]; db=b[:,None]-b[None,:]
    aa=np.sin(da/2)**2+np.cos(a[:,None])*np.cos(a[None,:])*np.sin(db/2)**2
    return (6371000*2*np.arcsin(np.sqrt(np.clip(aa,0,1)))).astype('float32')

def build_road_arrays(nodes,adj,distance,k=4):
    n=len(nodes)
    if adj.shape!=(n,n) or distance.shape!=(n,n):raise ValueError('Spatial source shape mismatch')
    if not np.array_equal(nodes.local_idx.to_numpy(),np.arange(n)):raise ValueError('nodes local_idx is not cache index order')
    for c in ('lat','lng','road_direction'):
        if c not in nodes:raise ValueError(f'Missing node field {c}')
    lat=pd.to_numeric(nodes.lat,errors='coerce').to_numpy(dtype=float)
    lon=pd.to_numeric(nodes.lng,errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():raise ValueError('Missing GPS in selected graph; no implicit index fallback')
    if ((np.abs(lat)>90)|(np.abs(lon)>180)).any():raise ValueError('GPS latitude/longitude invalid')
    road=nodes.road_direction.astype(str).to_numpy()
    if any(r.strip().lower() in ('','nan','none','unknown') for r in road):raise ValueError('Road direction missing')
    geo=haversine_m(lat,lon)
    adj_ok=(np.isfinite(adj)&(adj!=0))
    candidate=adj_ok|adj_ok.T
    np.fill_diagonal(candidate,False)
    progress=np.full(n,np.nan,dtype=float)
    for i,r in enumerate(road):
        direction=r.rsplit('__',1)[-1].strip().upper()
        if direction=='N':progress[i]=lat[i]
        elif direction=='S':progress[i]=-lat[i]
        elif direction=='E':progress[i]=lon[i]
        elif direction=='W':progress[i]=-lon[i]
        else:raise ValueError(f'Unsupported road_direction for node {i}: {r}')
    result={}
    for side in ('upstream','downstream'):
        ids=np.zeros((n,k),dtype=np.int64)
        valid=np.zeros((n,k),dtype=np.bool_)
        gps=np.zeros((n,k),dtype=np.float32)
        rd=np.zeros((n,k),dtype=np.float32)
        rdvalid=np.zeros((n,k),dtype=np.bool_)
        for i in range(n):
            same=(road==road[i])
            directed=(progress<progress[i]-1e-8) if side=='upstream' else (progress>progress[i]+1e-8)
            cand=np.flatnonzero(candidate[i]&same&directed&(geo[i]>0))
            if len(cand):
                # Deterministic geometric ranking avoids favoring -1 road distances.
                cand=cand[np.lexsort((cand,geo[i,cand]))][:k]
                m=len(cand)
                ids[i,:m]=cand;valid[i,:m]=True;gps[i,:m]=geo[i,cand]
                dv=distance[i,cand]
                usable=np.isfinite(dv)&(dv>0)
                rd[i,:m]=np.where(usable,dv,0)
                rdvalid[i,:m]=usable
        result[f'{side}_ids']=ids
        result[f'{side}_valid']=valid
        # Standardize to bounded, unit-consistent geometry; don't use road distances as weights.
        result[f'{side}_gps_log']=np.log1p(gps/1000).astype('float32')
        result[f'{side}_road_log']=np.log1p(rd/1000).astype('float32')
        result[f'{side}_road_valid']=rdvalid
    return result

def _paths(project_root,target,dataset):
    bench=Path(project_root)/'data/processed/traffident/base_supportbench'/target/dataset
    return [bench/'core/nodes.csv',bench/'graph/adjacency.npy',bench/'graph/distance.npy']

def graph_identity(project_root,target,dataset,cache_adj,k):
    paths=_paths(project_root,target,dataset)
    for file in (*paths,Path(cache_adj)):
        if not file.is_file():raise FileNotFoundError(f'Spatial graph source absent: {file}')
    return {'schema':GRAPH_SCHEMA,'target':target,'dataset':dataset,'neighbors':int(k),
            'files':{str(i):sha(v) for i,v in enumerate((*paths,Path(cache_adj)))}}

def _read_graph(project_root,target,dataset,cache_adj,k):
    paths=_paths(project_root,target,dataset)
    nodes=pd.read_csv(paths[0]);adj=np.load(paths[1]);dist=np.load(paths[2]);cached=np.load(cache_adj)
    if adj.shape!=cached.shape or not np.allclose(adj,cached,equal_nan=True):
        raise RuntimeError('Benchmark and ExperimentCache adjacency differ')
    return build_road_arrays(nodes,adj,dist,k)

def load_or_build_graph(project_root,target,dataset,cache_dir,out_dir,k=4):
    cache_adj=Path(cache_dir)/'adjacency.npy'
    identity=graph_identity(project_root,target,dataset,cache_adj,k)
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    meta=out/'graph_identity.json';file=out/'graph_context.npz'
    if meta.is_file() and file.is_file():
        if json.loads(meta.read_text())!=identity:raise RuntimeError('Graph source/version changed; use a NEW run directory')
        with np.load(file,allow_pickle=False) as z:arr={k:z[k].copy() for k in z.files}
        _validate(arr,int(np.load(cache_adj,mmap_mode='r').shape[0]),k)
        return arr
    if meta.exists() or file.exists():raise RuntimeError('Partial graph artifacts; remove TCRR graph subdir only and rerun')
    arr=_read_graph(project_root,target,dataset,cache_adj,k)
    _validate(arr,int(np.load(cache_adj,mmap_mode='r').shape[0]),k)
    tmp=out/'graph_context.partial.npz'
    np.savez_compressed(tmp,**arr)
    os.replace(tmp,file)
    info=out/'graph_validation.json'
    report={'version':'6.1.0','nodes':len(arr['upstream_ids']),
            'upstream_coverage':int(arr['upstream_valid'].any(1).sum()),
            'downstream_coverage':int(arr['downstream_valid'].any(1).sum()),
            'upstream_road_valid_edges':int(arr['upstream_road_valid'].sum()),
            'downstream_road_valid_edges':int(arr['downstream_road_valid'].sum()),
            'note':'All edge weights learned; missing road distances masked, no fallback affinity'}
    info.write_text(json.dumps(report,indent=2)+'\n')
    temp=out/'graph_identity.partial.json';temp.write_text(json.dumps(identity,indent=2));os.replace(temp,meta)
    print('V7_GRAPH_PASS '+json.dumps(report),flush=True)
    return arr

def _validate(arr,n,k):
    for side in ('upstream','downstream'):
        ids=arr[side+'_ids'];m=arr[side+'_valid'].astype(bool)
        if ids.shape!=(n,k) or m.shape!=(n,k):raise RuntimeError('Bad graph node/neighbour shape')
        if np.any(ids[m]>=n) or np.any(ids[m]<0):raise RuntimeError('Graph node index outside range')
        for attr in ('gps_log','road_log','road_valid'):
            if arr[side+'_'+attr].shape!=(n,k):raise RuntimeError(f'Bad graph {side} {attr} shape')
        if not np.isfinite(arr[side+'_gps_log']).all() or not np.isfinite(arr[side+'_road_log']).all():
            raise RuntimeError('Nonfinite graph edge features')
