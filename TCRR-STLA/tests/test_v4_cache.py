from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from tcrr_stla.v4.cache import FrozenBaseCache
from tcrr_stla.v4.trainer import calibrate_rejection

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stla=nn.Module()
        self.stla.base=nn.Module()
        self.stla.base.num_nodes=3
        self.stla.base.out_steps=2
        self.stla.model_dim=4
        self.stla.forward=lambda x,mask,tod,dow: (x[:,-1:].expand(-1,2,-1),
                                     torch.ones(x.shape[0],x.shape[-1],4))

class ToyDataset:
    def __init__(self,root):
        self.root=Path(root)
        for name,val in [('x_obs_norm',np.zeros((3,3,3),dtype='float32')),
                         ('history_observed',np.ones((3,3,3),dtype='bool')),
                         ('tod_hist',np.zeros((3,3),dtype='int64')),
                         ('dow_hist',np.zeros((3,3),dtype='int64'))]:
            np.save(self.root/f'train_{name}.npy',val)
    def __len__(self):return 3
    def __getitem__(self,i):
        return dict(x=torch.zeros(3,3),mask=torch.ones(3,3),
                    tod=torch.zeros(3,dtype=torch.long),dow=torch.zeros(3,dtype=torch.long))

def test_base_cache_is_fingerprinted_and_reused(tmp_path):
    ds=ToyDataset(tmp_path);ck=tmp_path/'backbone.pt';ck.write_bytes(b'ckpt')
    adapter=tmp_path/'adapter.pt';adapter.write_bytes(b'adapter')
    folder=tmp_path/'cache';model=ToyModel()
    a=FrozenBaseCache(folder,model,ds,'train',ck,adapter,'cpu')
    yp,z=a.fetch(torch.tensor([0,1]),'cpu')
    assert yp.shape==(2,2,3) and z.shape==(2,3,4)
    b=FrozenBaseCache(folder,model,ds,'train',ck,adapter,'cpu',skip_build=True)
    torch.testing.assert_close(b.fetch(torch.tensor([0]),'cpu')[0],yp[:1])
    adapter.write_bytes(b'changed')
    import pytest
    with pytest.raises(RuntimeError,match='stale'):
        FrozenBaseCache(folder,model,ds,'train',ck,adapter,'cpu',skip_build=True)

def test_risk_threshold_only_uses_validation_tail():
    q=10
    x=dict(prediction=np.ones((q,2,3)),base=np.zeros((q,2,3)),
           true=np.zeros((q,2,3)),valid=np.ones((q,2,3),bool),harm_probability=np.ones((q,2,3))*.8)
    th,d=calibrate_rejection(x)
    assert th<.8 and d['source']=='chronological_validation_last_20pct'
    x['true']=np.ones((q,2,3))
    th2,_=calibrate_rejection(x)
    assert th2==1.
