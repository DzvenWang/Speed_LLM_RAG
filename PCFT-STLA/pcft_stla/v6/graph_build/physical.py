"""Audited physical propagation candidates. Never read directed_neighbors.npz as graph truth."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

EDGE_FIELDS = ('raw_distance_valid', 'raw_distance_disagreement', 'geo_distance_scaled',
               'geo_valid', 'pm_distance_scaled', 'pm_valid', 'travel_forward',
               'disturbance_reverse', 'adjacency_strength', 'direction_quality')


def haversine_km(lat1,lon1,lat2,lon2):
    a1,a2=np.deg2rad([lat1,lat2]); dl=np.deg2rad(lat2-lat1); do=np.deg2rad(lon2-lon1)
    v=np.sin(dl/2)**2+np.cos(a1)*np.cos(a2)*np.sin(do/2)**2
    return 6371.0088*2*np.arcsin(np.sqrt(np.clip(v,0.,1.)))


def norm_dir(value):
    s=str(value).upper().replace('BOUND','').replace(' ','').strip()
    if s in ('N','NB','NORTH'):return 'N'
    if s in ('S','SB','SOUTH'):return 'S'
    if s in ('E','EB','EAST'):return 'E'
    if s in ('W','WB','WEST'):return 'W'
    return '?'


def build_physical(bench, max_geo_km=15., max_pm_gap_km=15.):
    from supportbench.utils import canonicalize_nodes
    nodes=canonicalize_nodes(bench.nodes)
    n=int(bench.N); adj=np.asarray(bench.adj,dtype=np.float64)
    if adj.shape!=(n,n):raise ValueError('Bad physical adjacency shape')
    if not np.isfinite(adj).all():raise ValueError('Nonfinite physical adjacency')
    if nodes['global_node_id'].astype(str).duplicated().any():raise ValueError('Duplicate global node identities')
    geo_lat=nodes['lat'].to_numpy(float);geo_lon=nodes['lng'].to_numpy(float)
    pm=nodes['postmile'].to_numpy(float)
    road=nodes['fwy'].astype(str).to_numpy()
    direction=[norm_dir(d) for d in nodes['direction']]
    dist_path=Path(bench.sg_dir)/'graph/distance.npy'
    raw=np.load(dist_path,mmap_mode='r') if dist_path.is_file() else None
    if raw is not None and raw.shape!=(n,n):raise ValueError('distance.npy shape invalid')
    edge_records=[];stats=dict(raw_distance_invalid=0,raw_distance_valid=0,ambiguous_pairs=0,
                              rejected_cross_road=0,rejected_distance=0,accepted_pairs=0,
                              unknown_distance_pairs=0,raw_distance_asymmetry=0)
    for i in range(n):
        for j in range(i+1,n):
            av=abs(adj[i,j]);bv=abs(adj[j,i]);
            if not (av>0 or bv>0):continue
            if road[i]!=road[j] or direction[i]=='?' or direction[i]!=direction[j]:
                stats['rejected_cross_road']+=1;continue
            valid_geo=np.isfinite(geo_lat[[i,j]]).all() and np.isfinite(geo_lon[[i,j]]).all()
            gd=float(haversine_km(geo_lat[i],geo_lon[i],geo_lat[j],geo_lon[j])) if valid_geo else 0.
            valid_geo=bool(valid_geo and gd>1e-6)
            valid_pm=bool(np.isfinite(pm[[i,j]]).all() and abs(pm[i]-pm[j])>1e-7)
            pdiff=1.609344*abs(float(pm[i]-pm[j])) if valid_pm else 0.
            # Geography is a geometric plausibility test, not an assumed wave speed.
            if (valid_geo and gd>max_geo_km) or (valid_pm and pdiff>max_pm_gap_km):
                stats['rejected_distance']+=1;continue
            # Latitude/longitude determines *local* direction; postmile is only a consistency cue.
            d=direction[i]
            coordinate=(geo_lat[j]-geo_lat[i]) if d in ('N','S') else (geo_lon[j]-geo_lon[i])
            direction_sign=1 if d in ('N','E') else -1
            if valid_geo and abs(coordinate)>1e-7:
                along=direction_sign*coordinate
                quality=1.0
            elif valid_pm:
                along=direction_sign*(pm[j]-pm[i]);quality=.5
            else:
                stats['ambiguous_pairs']+=1;continue
            if abs(along)<1e-10:stats['ambiguous_pairs']+=1;continue
            upstream,downstream=(i,j) if along>0 else (j,i)
            stats['accepted_pairs']+=1
            for src,dst,is_forward in ((upstream,downstream,True),(downstream,upstream,False)):
                rawvalue=float(raw[src,dst]) if raw is not None else float('nan')
                rawvalid=bool(np.isfinite(rawvalue) and rawvalue>0)
                if rawvalid:stats['raw_distance_valid']+=1
                else:stats['raw_distance_invalid']+=1
                other=float(raw[dst,src]) if raw is not None else float('nan')
                if rawvalid and np.isfinite(other) and other>0 and abs(rawvalue-other)>1e-6:
                    stats['raw_distance_asymmetry']+=1
                if not valid_geo and not valid_pm:stats['unknown_distance_pairs']+=1
                # Do not treat raw distance values as km: original unit is unverified.
                # No inverse-distance fallback; no Top-4 truncation.
                attrs=[float(rawvalid),float(rawvalid and np.isfinite(other) and other>0 and abs(rawvalue-other)>1e-6),min(gd/5.,10.) if valid_geo else 0.,float(valid_geo),
                       min(pdiff/5.,10.) if valid_pm else 0.,float(valid_pm),
                       float(is_forward),float(not is_forward),min(max(av,bv),10.),quality]
                edge_records.append((src,dst,attrs))
    if not edge_records:raise RuntimeError('No audited physical edges: inspect node metadata/direction/adjacency')
    edge_records.sort(key=lambda e:(e[0],e[1]))
    ei=np.asarray([(i,j) for i,j,_ in edge_records],dtype=np.int64).T.copy()
    ea=np.asarray([a for _,_,a in edge_records],dtype=np.float32)
    if not np.isfinite(ea).all():raise RuntimeError('Nonfinite physical edge features')
    node_ids=nodes['global_node_id'].astype(str).tolist()
    return ei,ea,dict(stats,physical_edges=int(ei.shape[1]),physical_nodes=n,
                      feature_schema=list(EDGE_FIELDS),node_ids=node_ids,
                      distance_raw_sha_missing=not dist_path.exists(),
                      distance_semantics='raw validity only; geo and postmile proxies separate')
