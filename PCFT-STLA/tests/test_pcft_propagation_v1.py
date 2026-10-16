"""Self-contained CPU unit tests for direction, observed support and future transfer."""
import tempfile
from pathlib import Path
import numpy as np
import torch
from pcft_stla.propagation import PropagationSupport,build_directional_graph


def _graph():
    n=8
    nb=torch.tensor([[(i+1)%n,(i+2)%n] for i in range(n)])
    return nb


def test_train_only_graph_and_support():
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)
        np.save(path/'train_x_full_norm.npy',np.random.default_rng(1).normal(size=(45,12,8)).astype('f'))
        nb=_graph();w,c=build_directional_graph(path,nb,path/'graph.npz',sampled_rows=32)
        assert w.shape==(3,8,2,6) and torch.isfinite(w).all()
        support=PropagationSupport(8,nb);support.install(w,c)
        mask=torch.zeros(2,12,8)
        assert torch.count_nonzero(support.support(mask,12))==0
        mask[:,-1]=1
        s=support.support(mask,12)
        assert (s[:,0]>0).any() and (s[:,6:]==0).all()
        assert support.signature(torch.randn(2,12,8),mask).shape==(2,12)
        out=support.transfer(torch.randn(2,4,12,8),torch.randn(2,4,12,8),12)
        assert out.shape==(2,4,12,8) and torch.isfinite(out).all()
