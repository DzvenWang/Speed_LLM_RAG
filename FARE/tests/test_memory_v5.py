import numpy as np
import torch
import pytest
from fare.memory import EpisodeMemory


def test_schema5_paired_persist(tmp_path):
    m,n,h,p=3,5,12,12
    x=torch.randn(m,h,n);mask=torch.ones_like(x,dtype=torch.bool)
    y=torch.randn(m,p,n);valid=torch.ones_like(y,dtype=torch.bool)
    base=torch.randn(m,p,n);failure=torch.randn(m,p,n)
    bank=EpisodeMemory(x,mask,y,valid,torch.arange(m)*100,torch.arange(m)*100+11,
                       base_forecast=base,failure_delta=failure,metadata={'residual_provenance':'external_oos_paired'})
    bank.save(tmp_path/'store')
    got=EpisodeMemory.load(tmp_path/'store')
    assert got.metadata['schema_version']==6
    assert got.metadata['residual_provenance']=='external_oos_paired'
    assert torch.allclose(got.failure_delta,failure,atol=.004)
    assert got.failure_valid.all()
    assert got.valid_for(torch.tensor([400]),'cpu',purge=12).all()
    assert not got.valid_for(torch.tensor([5]),'cpu').any()


def test_legacy_schema_fails_closed(tmp_path):
    import json
    p=tmp_path/'old';p.mkdir();(p/'manifest.json').write_text(json.dumps({'format':'fare4_memmap'}))
    with pytest.raises(RuntimeError,match='legacy'):
        EpisodeMemory.load(p)


def test_oof_cutoff_validation(monkeypatch,tmp_path):
    from fare import memory
    class Fake:
        def __init__(self,cache,split):
            import pandas as pd
            self.meta=pd.DataFrame({'window_start':[100,200,300]})
            self.h=12;self.p=12
            self.arr={'x_obs_norm':np.random.randn(3,12,2),
                      'history_observed':np.ones((3,12,2),dtype=bool),
                      'y_norm':np.zeros((3,12,2)),
                      'future_valid':np.ones((3,12,2),dtype=bool)}
    monkeypatch.setattr('fare.protocol.SupportDataset',Fake)
    f=tmp_path/'oof.npz';
    np.savez(f,window_start=np.array([100,200,300]),
             trained_until=np.array([20,200,250]),
             prediction=np.zeros((3,12,2)),prediction_complete=np.ones((3,12,2)))
    with pytest.raises(RuntimeError,match='schema-6'):
        memory.from_support_cache(tmp_path,None,3,'external_oos_paired',f)
