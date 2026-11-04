"""TRAIN-only optional multi-severity sensing, using canonical SupportBench raw data and sensing layouts.

The formal validation and evaluation sets are NEVER transformed. This is a distinct
training protocol from v6 fixed-s40 and should be reported separately in papers.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from pcft_stla.v6.data import age_sequence

class MixedTrainMasks(Dataset):
    """Rebuild INPUT histories from raw TRAIN traffic, never from held-out future targets.

    Distinct training views are chosen deterministically from a schedule of
    severities and sensing families. Original raw validity remains authoritative.
    """
    def __init__(self,base_dataset,bench,severities=(20,40,60,90),families=None,seed=10):
        self.base=base_dataset
        self.bench=bench
        self.meta=base_dataset.meta
        self.target_idx=base_dataset.target_idx
        self.severities=list(severities)
        self.families=list(families or ('entry_random','sensor_outage','temporal_outage',
                                       'regional_outage','directional_outage'))
        self.seed=int(seed)
        if not self.severities or not self.families:raise ValueError('Nonempty severity/family schedule required')
        if self.target_idx!=bench.channel_names.index(base_dataset.target):
            raise ValueError('Mixed target index vs SupportBench mismatch')
    def __len__(self):return len(self.base)
    def __getitem__(self,index):
        from supportbench.sensing import materialize_mask
        from supportbench.utils import stable_seed
        sample=self.base[index]
        start=int(self.meta.iloc[index].window_start)
        H=int(self.bench.H);N=int(self.bench.N)
        severity=int(self.severities[index%len(self.severities)])
        family=self.families[(index//len(self.severities))%len(self.families)]
        seed=stable_seed(Path(self.bench.sg_dir).name,'train',start,family,severity,index,self.seed,base=self.seed)
        spec={'family':family,'severity':severity/100.,'seed':seed,'distribution':'id',
              'temporal_node_fraction':0.75}
        hist=np.asarray(self.bench.traffic[start:start+H],np.float32)
        valid=np.asarray(self.bench.valid[start:start+H],dtype=bool)
        if hist.shape!=(H,N,len(self.bench.channel_names)):
            raise ValueError('TRAIN raw history shape mismatch')
        artificial=materialize_mask(spec,H,N,self.bench.sg_dir,valid[...,self.target_idx])
        observed=valid & (~artificial[...,None])
        normalized=self.bench.normalize(hist)
        x=np.where(observed,normalized,0.).astype(np.float32)
        if not np.isfinite(x).all():raise ValueError('Nonfinite multi-mask TRAIN features')
        # Only input features are rematerialized; future labels remain the base TRAIN sample.
        sample.update(x_multi=torch.from_numpy(x),
                      observed_mask=torch.from_numpy(observed.astype(np.float32)),
                      missing_age=torch.from_numpy(age_sequence(observed).astype(np.float32)),
                      x_target=torch.from_numpy(x[...,self.target_idx].copy()),
                      direct_mask=torch.from_numpy(artificial.any(0).copy()))
        return sample
