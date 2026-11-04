"""Block bootstrap over chronological original time coordinates (not independent entries)."""
from __future__ import annotations
import numpy as np


def block_bootstrap(window_rows,block_steps=288,repeats=400,seed=10):
    if not window_rows:return {'status':'no_windows'}
    if repeats<2:raise ValueError('repeats must be >=2')
    grouping={}
    for row in window_rows:
        key=int(row['window_start'])//int(block_steps)
        g=grouping.setdefault(key,dict(base=0.,final=0.,count=0))
        n=int(row['valid_positions'])
        g['base']+=row['base_mae']*n
        g['final']+=row['controlled_mae']*n
        g['count']+=n
    blocks=list(grouping.values());nb=len(blocks)
    if nb<5:
        return {'status':'insufficient_independent_blocks','block_count':nb,
                'block_steps':int(block_steps),'note':'Do not treat windows as independent for uncertainty estimation'}
    rng=np.random.default_rng(seed)
    matrix=np.array([[b['base'],b['final'],b['count']] for b in blocks],np.float64)
    estimates=[]
    for _ in range(repeats):
        sampled=matrix[rng.integers(0,nb,size=nb)].sum(axis=0)
        if sampled[0]>0:estimates.append(100*(sampled[0]-sampled[1])/sampled[0])
    return {'status':'ok','block_count':nb,'block_steps':int(block_steps),'bootstrap_samples':len(estimates),
            'improvement_pct_95ci_low':float(np.percentile(estimates,2.5)),
            'improvement_pct_95ci_high':float(np.percentile(estimates,97.5)),
            'method':'resample nonoverlapping chronological time blocks; conditional on selected windows & fixed trained model'}
