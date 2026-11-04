import numpy as np
import pandas as pd
import torch
from pcft_stla.v7.data import MixedTrainMasks

class FakeBase:
    target_idx=0
    target='speed'
    meta=pd.DataFrame({'window_start':[0,12,24,36]})
    def __len__(self):return 4
    def __getitem__(self,i):
        return {'y_norm':torch.ones(12,4),'future_valid':torch.ones(12,4,dtype=torch.bool),
            'direct_mask':torch.zeros(4,dtype=torch.bool),
            'x_multi':torch.zeros(12,4,3),'observed_mask':torch.zeros(12,4,3),
            'missing_age':torch.zeros(12,4,3),'x_target':torch.zeros(12,4)}

class FakeBench:
    H=12;N=4
    sg_dir='not_needed_for_sensor_outage'
    channel_names=['speed','flow','occupancy']
    traffic=np.arange(120*4*3,dtype=np.float32).reshape(120,4,3)
    valid=np.ones((120,4,3),dtype=bool)
    def normalize(self,hist):return hist/100.

def test_mixed_train_masks_reconstruct_only_history_and_preserve_train_labels():
    base=FakeBase();ds=MixedTrainMasks(base,FakeBench(),[20,90],['sensor_outage'],seed=10)
    x=ds[1]
    assert x['observed_mask'].shape==(12,4,3)
    assert x['observed_mask'][0].sum()<12
    assert torch.equal(x['y_norm'],base[1]['y_norm'])
    assert torch.equal(x['future_valid'],base[1]['future_valid'])
    assert (x['x_multi'][x['observed_mask']==0]==0).all()
    assert torch.allclose(x['x_target'],x['x_multi'][...,0])
