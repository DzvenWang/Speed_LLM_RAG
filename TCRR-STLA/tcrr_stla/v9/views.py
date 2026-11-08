"""Deterministic paired sensing failures. Source-valid observation is never fabricated."""
from __future__ import annotations
import hashlib
import numpy as np

FAMILIES = ('entry_random','sensor_outage','temporal_outage','regional_outage','directional_outage')

def mask_sha(mask: np.ndarray) -> str:
    b=np.ascontiguousarray(mask, dtype=np.bool_)
    return hashlib.sha256(b.tobytes()+str(b.shape).encode()).hexdigest()

def make_mask(source_valid: np.ndarray, *, family: str, severity: int, seed: int,
              region: np.ndarray | None = None) -> np.ndarray:
    """1=visible. region: optional node region labels ordered as dataset nodes.

    Directional outages require supplied ordered physical-node positions (region labels
    may be ordered physical groups); otherwise explicit fallback to contiguous node
    blocks is forbidden. No use of node-index proximity as physical road distance.
    """
    source_valid=np.asarray(source_valid,bool)
    if source_valid.ndim != 2:raise ValueError('Expected [L,N]')
    if family not in FAMILIES:raise ValueError(family)
    if not (1<=severity<=99):raise ValueError('severity must be 1..99')
    l,n=source_valid.shape; rng=np.random.default_rng(seed); fraction=severity/100.0
    view=np.ones((l,n),dtype=bool)
    if family=='entry_random':
        view = rng.random((l,n))>=fraction
    elif family=='sensor_outage':
        bad=rng.choice(n,max(1,round(n*fraction)),replace=False)
        view[:,bad]=False
    elif family=='temporal_outage':
        length=max(1,round(l*fraction)); start=int(rng.integers(0,l-length+1))
        view[start:start+length]=False
    else:
        if region is None:
            raise ValueError(f'{family} requires physically grounded region labels')
        region=np.asarray(region)
        if region.shape!=(n,):raise ValueError('region must be [N]')
        distinct=np.unique(region)
        # Region outages select whole physical regions; directional requires
        # direction/chain-specific region encoding from graph metadata.
        if family=='directional' and not np.issubdtype(region.dtype,np.number):
            raise ValueError('directional region labels must encode physical chain order')
        order=rng.permutation(len(distinct))
        if family=='directional':
            order=np.arange(len(distinct)) if int(rng.integers(2))==0 else np.arange(len(distinct)-1,-1,-1)
        selected=[]
        for ind in order:
            selected.extend(np.flatnonzero(region==distinct[ind]).tolist())
            if len(selected)>=max(1,round(n*fraction)):break
        view[:,selected]=False
    return view & source_valid

def view_seed(seed:int,query_id:int,view_id:int)->int:
    raw=f'{seed}:{query_id}:{view_id}:tcrr9'.encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],'little')

def paired_mask(source_valid,query_id,seed,view_id,region=None):
    if view_id==0:raise ValueError('view 0 is the official mask, not generated')
    family=FAMILIES[(query_id+view_id+seed)%len(FAMILIES)]
    if region is None and family in ('regional_outage','directional_outage'):
        family=('sensor_outage' if family=='regional_outage' else 'temporal_outage')
    severity=(20,40,60,90)[(query_id//len(FAMILIES)+view_id+seed)%4]
    m=make_mask(source_valid,family=family,severity=severity,
                seed=view_seed(seed,query_id,view_id),region=region)
    return m,family,severity
