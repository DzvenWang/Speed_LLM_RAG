"""Reproducible validation sampling without looking at future labels."""
from __future__ import annotations
import hashlib
import numpy as np

def choose_indices(dataset, count:int, seed:int=10):
    n=len(dataset)
    if count<=0 or count>=n:return np.arange(n,dtype=np.int64)
    rng=np.random.default_rng(int(seed))
    # Evenly spread across ordered validation windows; sampling never sees y.
    bins=np.array_split(np.arange(n,dtype=np.int64),count)
    selected=np.array([rng.choice(b) for b in bins],dtype=np.int64)
    return np.sort(selected)

def sample_manifest(ds,indices):
    names=ds.meta.columns
    rows=[]
    for ix in indices:
        row=ds.meta.iloc[int(ix)]
        data={'index':int(ix)}
        for col in ('window_start','view_id','family','severity'):
            if col in names:
                value=row[col]
                data[col]=int(value) if col in ('window_start','severity') and str(value).isdigit() else str(value)
        rows.append(data)
    return rows

def manifest_sha(rows):
    import json
    return hashlib.sha256(json.dumps(rows,sort_keys=True).encode('utf-8')).hexdigest()
