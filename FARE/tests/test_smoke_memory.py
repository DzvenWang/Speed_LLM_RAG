"""Regression: training-smoke must retain legal TRAIN retrieval candidates."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from fare.config import load_config
from fare.memory import TrainMemory
from fare import data as fare_data


def test_train_memory_spread_and_formal_unchanged():
    rows, history, nodes = 240, 12, 9
    starts = np.arange(rows,dtype=np.int64)*12
    class StubSplit:
        def __init__(self,*args,**kwargs):
            self.meta=pd.DataFrame({'window_start':starts})
            self.arr={
                'x_full_norm':np.zeros((rows,history,nodes),dtype=np.float32),
                'history_observed':np.ones((rows,history,nodes),dtype=bool),
                'artificial_missing':np.zeros((rows,history,nodes),dtype=bool),
                'tod_hist':np.zeros((rows,history),dtype=np.int64),
                'dow_hist':np.zeros((rows,history),dtype=np.int64),
            }
    original=fare_data.SupportSplit
    try:
        fare_data.SupportSplit=StubSplit
        cfg=load_config(smoke=True)
        assert cfg['memory']['sampling']=='spread'
        memory=TrainMemory.from_cache('unused',cfg['memory']['max_rows'],sampling=cfg['memory']['sampling'])
        assert len(memory.x)==64
        # Includes distant training-history evidence, unlike the old first-16 policy.
        q=torch.as_tensor(starts[:8])
        legal=(q[:,None]-memory.starts[None,:]).abs()>=24
        assert legal.any(dim=1).all()
        assert int(memory.starts.max())==int(starts.max())
        formal=load_config()
        assert formal['memory']['sampling']=='stratified_temporal'
        assert formal['memory']['max_rows']==50000
        print('FARE_SMOKE_MEMORY_SPREAD_PASS legal_candidates=',legal.sum(-1).tolist())
    finally:
        fare_data.SupportSplit=original


if __name__=='__main__':
    test_train_memory_spread_and_formal_unchanged()