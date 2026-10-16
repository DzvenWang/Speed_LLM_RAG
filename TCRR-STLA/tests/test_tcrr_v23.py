import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from torch import nn
from tcrr_stla.forecast_pool import select_unique,prepare_forecast_pool
from tcrr_stla.data import RetrievalCache
from tcrr_stla.evidence import shortlist
from tcrr_stla.wrong_evidence import legal_wrong_indices


def test_unique_retrieval_pool():
    ids=np.array([[3,3,4,8,4,9,5]],np.int32)
    hist=np.array([[.99,.98,.97,.96,.94,.92,.90]],np.float32)
    ci,cs,cc=select_unique(ids,hist,np.ones_like(hist),np.zeros_like(hist),5)
    assert ci.tolist()==[[3,4,8,9,5]]
    assert len(set(ci[0]))==5


def test_forecast_pool_fingerprint_and_index(tmp_path,monkeypatch):
    root=tmp_path;formal=root/'formal';formal.mkdir()
    M,Q,H,P,N=30,24,12,12,4
    rng=np.random.default_rng(3)
    np.save(formal/'train_y_norm.npy',rng.normal(size=(M,P,N)).astype('f4'))
    np.save(root/'train_x_obs_norm.npy',rng.normal(size=(Q,H,N)).astype('f4'))
    np.save(root/'train_history_observed.npy',(rng.random((Q,H,N))>.5).astype('f4'))
    np.save(root/'train_tod_hist.npy',np.zeros((Q,H),dtype='i8'))
    np.save(root/'train_dow_hist.npy',np.zeros((Q,H),dtype='i8'))
    np.savez(root/'raw.npz',indices=np.broadcast_to(np.arange(12)[None,None,:],(Q,3,12)).copy(),
             scores=np.broadcast_to(np.linspace(.8,.1,12)[None,None,:],(Q,3,12)).copy(),
             coverage=np.ones((Q,3),dtype='f4'))
    ck=root/'ckpt.pt';ck.write_bytes(b'fake-model-hash')
    import tcrr_stla.stla as stla
    class MockTeacher(nn.Module):
        def __init__(self,*args):super().__init__()
        def forward(self,x,mask,tod,dow):
            return (x*mask).mean(1)[:,None].expand(-1,12,-1),None
    monkeypatch.setattr(stla,'load_pretrained_stla',lambda *args:(nn.Identity(),{}))
    monkeypatch.setattr(stla,'MaskAwareSTLAformer',MockTeacher)
    cfg={'project':{'root':str(root)},'retrieval':{'final_candidates':12,'history_similarity_weight':.55,'forecast_pool_batch_size':5}}
    path=prepare_forecast_pool(cfg,formal,root,'train',root/'raw.npz',ck,torch.device('cpu'),smoke=True)
    cache=RetrievalCache(path)
    a,sc,cov=shortlist(cache,np.arange(4),4,torch.device('cpu'))
    assert a.shape==(4,12) and sc.shape==cov.shape==a.shape
    assert all(len(set(row))==12 for row in a.tolist())
    again=prepare_forecast_pool(cfg,formal,root,'train',root/'raw.npz',ck,torch.device('cpu'),smoke=True)
    assert again==path


def test_cross_history_invariant_to_batching():
    starts=np.arange(80)*30
    whole=legal_wrong_indices([90,120,150],12,starts,24,'cpu')
    parts=torch.cat([legal_wrong_indices([90],12,starts,24,'cpu'),
                     legal_wrong_indices([120,150],12,starts,24,'cpu')],0)
    assert torch.equal(whole,parts)
    assert all(abs(starts[int(whole[b,k])]-s)>=24 for b,s in enumerate((90,120,150)) for k in range(12))

def test_direct_masked_audit_all_horizons():
    from scripts.evaluate_retrieval import record
    y=np.zeros((2,12,3),dtype='f4')
    pred=np.ones_like(y)
    direct=np.array([[1,0,1],[0,1,0]],dtype=bool)
    valid=np.ones_like(y,dtype=bool)&direct[:,None,:]
    row=record('Audit',40,'direct_masked',11,None,y,valid,pred,pred,pred,pred)
    assert row['observations']==3 and row['horizon']==12
