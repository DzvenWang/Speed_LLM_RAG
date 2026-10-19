"""End-to-end model -> paired OOF Memory -> teacher -> train -> four severities Evaluate.
No synthetic fallback exists in production; this test monkeypatches only the
external Base_SupportBench/Backbone data producers and metric dependency.
"""
import torch
import numpy as np
import pandas as pd
from fare import pipeline
from fare import evaluation
from conftest import DummyBackbone


def test_complete_pipeline_with_paired_oof(tmp_path,monkeypatch):
    n,h,p=5,12,12
    cache=tmp_path/'cache';cache.mkdir()
    rng=np.random.default_rng(10)
    for split,count,offset in [('train',8,100),('val',3,1000),('test',3,2000)]:
        x=rng.standard_normal((count,h,n)).astype('float32')
        y=x.mean(1)[:,None,:]+.1*rng.standard_normal((count,p,n)).astype('float32')
        fields={'x_obs_norm':x,'x_full_norm':x.copy(),'history_observed':np.ones_like(x,dtype=bool),
                'artificial_missing':np.zeros_like(x,dtype=bool),'y_norm':y,'y_raw':y.copy(),
                'future_valid':np.ones_like(y,dtype=bool),'direct_mask':np.zeros((count,n),dtype=bool),
                'tod_hist':np.zeros((count,h),dtype=np.int16),'dow_hist':np.zeros((count,h),dtype=np.int16)}
        for name,v in fields.items():np.save(cache/f'{split}_{name}.npy',v)
        pd.DataFrame({'window_start':offset+np.arange(count)*30}).to_csv(cache/f'{split}_manifest.csv',index=False)
    adj=np.eye(n,dtype='float32')
    for j in range(n-1):adj[j,j+1]=adj[j+1,j]=1
    np.save(cache/'adjacency.npy',adj)
    ck=tmp_path/'stla.pt';torch.save({'model':'mock'},ck)
    oof=tmp_path/'paired.npz';st=100+np.arange(8)*30
    from fare.oof import row_keys
    metadata=pd.read_csv(cache/'train_manifest.csv')
    np.savez(oof,row_id=np.arange(8),episode_key=row_keys(metadata),window_start=st,
             fold_id=np.zeros(8,dtype=np.int16),trained_until=np.full(8,50,dtype=np.int64),
             prediction=np.zeros((8,p,n),dtype='float32'),
             prediction_complete=np.full((8,p,n),.03,dtype='float32'))
    # The model/checkpoint producer path is unit-tested separately; fixture supplies a
    # deterministic, structurally valid mock OOF source for the full downstream model.
    from fare import oof as oof_module
    real_verify=oof_module.verify_paired
    monkeypatch.setattr(oof_module,'verify_paired',lambda path,ds,strict=True:real_verify(path,ds,strict=False))
    monkeypatch.setattr(pipeline,'root',lambda:tmp_path)
    monkeypatch.setattr(pipeline,'cache_dir',lambda *a,**k:cache)
    monkeypatch.setattr(pipeline,'backbone_checkpoint',lambda *a,**k:ck)
    monkeypatch.setattr(pipeline,'load_stla',lambda *a,**k:DummyBackbone(n))
    monkeypatch.setattr(pipeline,'directed_metadata',lambda *a,**k:None)
    def metrics(pred,true,valid,direct):
        x=(pred-true)[valid]
        return {'overall_MAE':float(np.abs(x).mean()),'overall_RMSE':float(np.sqrt((x*x).mean())),
                'overall_MAPE':None,'masked_MAE':None,'masked_RMSE':None}
    monkeypatch.setattr(evaluation,'canonical_metric_bundle',lambda:metrics)
    c={'experiment':{'dataset':'TraffiDent_Net_06','target':'speed','seed':10,'run_tag':'formal',
             'train_limit':6,'val_limit':2,'test_limit':2,'sample_strategy':'spread',
             'severities':[20,40,60,90]},
       'model':{'hidden_dim':8,'horizon':12,'alpha_max':1.25},
       'backbone':{'inject_layers':[4,5]},
       'traffic':{'region_size':3,'neighbor_count':2},
       'memory':{'max_episodes':8,'residual_mode':'external_oos_paired',
                 'require_oos_for_formal':True,'purge_steps':12,'build_batch_size':2},
       'retrieval':{'coarse_k':4,'shortlist_k':3,'final_k':2},
       'teacher':{'max_queries':4,'batch_size':2,'candidates_per_query':3},
       'training':{'batch_size':2,'workers':0,'lr':1e-3,'weight_decay':0.,'grad_clip':3.,
                   'masked_weight':.2,'tail_weight':.02,'utility_weight':.3,
                   'replay_interval':1,'replay_batch':2,'log_every':100,
                   'stage_epochs':{'evidence':1,'retrieval':1,'joint':1,'risk':1}},
       'evaluation':{'auto_evaluate':True,'include_val':False},
       'analysis':{'enabled':False}}
    run=pipeline.run(c,torch.device('cpu'),run_dir=tmp_path/'run',
                     oos_forecast_file=oof,analysis=False)
    assert (run/'best.pt').is_file()
    assert (run/'evaluation/performance_summary.csv').is_file()
    assert (run/'pipeline_complete.json').is_file()
    assert (tmp_path/'FARE_cache/speed/TraffiDent_Net_06/memory_v502_10_8/manifest.json').is_file()
