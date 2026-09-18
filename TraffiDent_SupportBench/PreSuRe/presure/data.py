from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

class CacheSplit:
    def __init__(self,root,split):
        r=Path(root); self.root=r; self.split=split
        def mm(n): return np.load(r/f'{split}_{n}.npy',mmap_mode='r')
        self.x=mm('x_obs_norm'); self.obs=mm('history_observed'); self.y=mm('y_raw'); self.valid=mm('future_valid')
        self.direct=mm('direct_mask'); self.support=mm('support_mask'); self.tod=mm('tod_hist'); self.dow=mm('dow_hist')
        self.meta=pd.read_csv(r/f'{split}_manifest.csv')
    def __len__(self): return len(self.x)

def cache_stats(cache_root):
    import json
    j=json.loads((Path(cache_root)/'cache_manifest.json').read_text())
    # all splits share normalization statistics
    first=next(iter(j['splits'].values()))
    return float(first['mean']),float(first['std'])
