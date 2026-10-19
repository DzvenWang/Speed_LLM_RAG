import numpy as np
import torch
from pcft_stla.propagation_v2 import build_directional_graph

def test_local_regime_graph_train_only_and_cache(tmp_path):
    rng=np.random.default_rng(16)
    train=tmp_path/'formal';train.mkdir()
    x=rng.normal(size=(35,12,5)).astype('f4')
    np.save(train/'train_x_full_norm.npy',x)
    neighbours=torch.tensor([[1,2],[0,2],[1,3],[2,4],[3,0]])
    fn=tmp_path/'graph.npz'
    w,c=build_directional_graph(train,neighbours,fn,lags=(1,2,3),sampled_rows=20)
    assert w.shape==c.shape==(3,5,2,3)
    assert torch.isfinite(w).all()
    w2,c2=build_directional_graph(train,neighbours,fn,lags=(1,2,3),sampled_rows=20)
    assert torch.equal(w,w2) and torch.equal(c,c2)
