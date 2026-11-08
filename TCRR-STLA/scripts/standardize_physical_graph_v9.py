#!/usr/bin/env python3
"""Convert node-aligned directed road graphs into v9 graph NPZ.

Edges oriented as (target_node_index, source_neighbour_index), not source->target.
Never guess orientation from node index or a CSV row position.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

def as_adj(z,prefix,n):
    if prefix in z:
        a=np.asarray(z[prefix],np.float32)
        if a.shape!=(n,n):raise ValueError(f'{prefix} must be NxN')
        return a
    for key in (prefix+'_edges',prefix+'_indices'):
        if key in z:
            arr=np.asarray(z[key]);a=np.zeros((n,n),np.float32)
            if key.endswith('_edges'):
                if arr.ndim!=2 or arr.shape[1]!=2:raise ValueError(key+' must be [E,2]')
                for target,source in arr:
                    if not (0<=target<n and 0<=source<n):raise ValueError('out-of-range edge')
                    a[int(target),int(source)]=1
            else:
                if arr.ndim!=2 or arr.shape[0]!=n:raise ValueError(key+' must be [N,K]')
                for i,neighbors in enumerate(arr):
                    for source in neighbors:
                        if source<0:continue
                        if source>=n:raise ValueError('out-of-range neighbor')
                        a[i,int(source)]=1
            return a
    raise ValueError(f'No {prefix} adjacency/edge/index arrays; known keys: {list(z.files)}')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--node-ids',type=Path,required=True,help='actual dataset-ordered sensor IDs NPY')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--orientation-confirmed',action='store_true',
        help='Confirm every input edge means target row receives from source column')
    a=p.parse_args()
    if not a.orientation_confirmed:raise ValueError('Explicit --orientation-confirmed is required')
    ids=np.load(a.node_ids,allow_pickle=False);n=len(ids)
    if len(np.unique(ids))!=n:raise ValueError('Duplicate node IDs')
    with np.load(a.input,allow_pickle=False) as z:
        if 'node_ids' in z and not np.array_equal(z['node_ids'].astype(str),ids.astype(str)):
            raise ValueError('Input graph node_id order conflicts with dataset')
        up=as_adj(z,'upstream',n);down=as_adj(z,'downstream',n)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,node_ids=ids,upstream=up,downstream=down)
    print('V9_GRAPH_STANDARDIZED',a.output,'N',n,'up_edges',int((up!=0).sum()),
          'down_edges',int((down!=0).sum()))
if __name__=='__main__':main()
