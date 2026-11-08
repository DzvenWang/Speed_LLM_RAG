import numpy as np
import torch
from tcrr_stla.traffic_neighbors import train_only_neighbours
from tcrr_stla.traffic_v2 import FailureConditionedTraffic

def test_train_only_support_graph_and_all_missing_nodes(tmp_path):
    rng=np.random.default_rng(7)
    histories=rng.normal(size=(28,12,6)).astype('f4')
    histories[:,:,2]=histories[:,:,0]*.8+rng.normal(0,.05,(28,12))
    np.save(tmp_path/'train_x_full_norm.npy',histories)
    nbr=train_only_neighbours(tmp_path,3)
    assert nbr.shape==(6,3)
    module=FailureConditionedTraffic(12)
    module.install_neighbors(nbr)
    x=torch.randn(2,12,6)
    m=torch.ones_like(x);m[0,:,0]=0
    c=torch.randn(2,4,12,6)
    r=torch.randn(2,4,12,6)
    out=module(x,m,c,torch.randn(2,12,6),r,r)
    assert out['expected_gain'].shape==(2,4,12,6)
    assert torch.isfinite(out['expected_gain']).all()
