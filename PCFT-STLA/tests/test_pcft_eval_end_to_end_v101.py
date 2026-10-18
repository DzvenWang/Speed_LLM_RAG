"""End-to-end CPU evaluation regression for safe batches vs old training batches."""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from pcft_stla.config import load_config
from pcft_stla.model import PCFTSTLA
from pcft_stla.data import FullTrainRepository, SupportSplit
from pcft_stla.memory import build_memory_index
from pcft_stla.seasonal import train_future_reference
from pcft_stla.trainer import predict, make_loader
from pcft_stla.runtime_safety import safe_evaluation_config, predict_with_retry
from test_pcft_v2 import STLA


def test_recovery_predict_batch_independent(tmp_path):
    torch.manual_seed(123)
    rng=np.random.default_rng(123)
    M,B,H,P,N=32,4,12,12,7
    np.save(tmp_path/'train_x_full_norm.npy',rng.normal(size=(M,H,N)).astype('f4'))
    np.save(tmp_path/'train_y_norm.npy',rng.normal(size=(M,P,N)).astype('f4'))
    np.save(tmp_path/'train_tod_hist.npy',np.zeros((M,H),dtype='i8'))
    np.save(tmp_path/'train_dow_hist.npy',np.zeros((M,H),dtype='i8'))
    pd.DataFrame({'window_start':np.arange(M)*30}).to_csv(tmp_path/'train_manifest.csv',index=False)
    for suffix,data in {
        'x_obs_norm':rng.normal(size=(B,H,N)).astype('f4'),
        'history_observed':np.ones((B,H,N),dtype='f4'),
        'y_norm':rng.normal(size=(B,P,N)).astype('f4'),
        'y_raw':rng.normal(size=(B,P,N)).astype('f4'),
        'future_valid':np.ones((B,P,N),dtype='bool'),
        'direct_mask':np.ones((B,N),dtype='bool'),
        'tod_hist':np.zeros((B,H),dtype='i8'),
        'dow_hist':np.zeros((B,H),dtype='i8'),
    }.items():
        np.save(tmp_path/f'test_{suffix}.npy',data)
    pd.DataFrame({'window_start':np.arange(B)*30+10000,'family':['entry_random']*B}).to_csv(tmp_path/'test_manifest.csv',index=False)
    repo=FullTrainRepository(tmp_path)
    ds=SupportSplit(tmp_path,'test')
    cfg=load_config()
    cfg['retrieval'].update(top_k=2,regional_top_k=2,graph_neighbors=3)
    model=PCFTSTLA(STLA(),N,H,P,cfg)
    model.set_graph(torch.tensor([[(i+j+1)%N for j in range(3)] for i in range(N)]))
    model.set_future_reference(train_future_reference(repo,P,N))
    model.eval()
    idx=build_memory_index(model.memory_encoder,repo,'cpu',16)
    safe=safe_evaluation_config(cfg,8,0,False)
    b4=predict(model,make_loader(ds,4,False,safe),idx,repo,torch.device('cpu'),0.,1.,'full')
    b1,used=predict_with_retry(model,ds,idx,repo,torch.device('cpu'),0.,1.,
                        'full',False,safe,make_loader,predict,1)
    assert used==1 and b4['prediction'].shape==(B,P,N)
    np.testing.assert_allclose(b4['prediction'],b1['prediction'],rtol=1e-5,atol=1e-5)
    np.testing.assert_allclose(b4['base'],b1['base'],rtol=1e-5,atol=1e-5)
