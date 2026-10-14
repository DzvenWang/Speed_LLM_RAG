"""End-to-end FARE-3 pipeline on a synthetic SupportBench-shaped fixture.

External data provider and STLAformer are mocked only in this integration test.
"""
import numpy as np,torch,pandas as pd
from pathlib import Path

def test_pipeline_train_evaluate_analyze(fixture_data,tmp_path,monkeypatch):
    import fare.pipeline as pipeline
    import fare.evaluation as evaluation
    from conftest import DummyBackbone
    _,_,b=fixture_data
    n=6;h=12;p=12
    cache=tmp_path/'cache';cache.mkdir()
    rng=np.random.default_rng(10)
    for split,count,offset in [('train',8,0),('val',3,250),('test',3,350)]:
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
    stub_ck=tmp_path/'mock_stla.pt';torch.save({'test':True},stub_ck)
    monkeypatch.setattr(pipeline,'root',lambda:tmp_path)
    monkeypatch.setattr(pipeline,'cache_dir',lambda *args,**kwargs:cache)
    monkeypatch.setattr(pipeline,'backbone_checkpoint',lambda *args,**kwargs:stub_ck)
    monkeypatch.setattr(pipeline,'load_stla',lambda *args,**kwargs:DummyBackbone(n))
    monkeypatch.setattr(pipeline,'directed_metadata',lambda *args,**kwargs:None)
    def metrics(pred,true,valid,direct):
        d=(pred-true)[valid]
        return {'overall_MAE':float(np.abs(d).mean()),'overall_RMSE':float(np.sqrt(np.mean(d*d))),
                'overall_MAPE':float('nan'),'masked_MAE':float('nan'),'masked_RMSE':float('nan')}
    monkeypatch.setattr(evaluation,'canonical_metric_bundle',lambda:metrics)
    c={'experiment':{'dataset':'TraffiDent_Net_06','target':'speed','seed':10,'run_tag':'pilot',
                     'train_limit':6,'val_limit':3,'test_limit':2,'sample_strategy':'spread',
                     'severities':[20,40,60,90]},
       'model':{'hidden_dim':8,'horizon':12,'alpha_max':1.25},
       'backbone':{'inject_layers':[4,5]},
       'traffic':{'region_size':3,'neighbor_count':2},
       'memory':{'max_episodes':8,'residual_mode':'checkpoint_in_sample','require_oos_for_formal':False,
                 'purge_steps':0,'build_batch_size':2},
       'retrieval':{'coarse_k':4,'shortlist_k':3,'final_k':2},
       'teacher':{'max_queries':4,'batch_size':2},
       'training':{'batch_size':2,'workers':0,'lr':1e-3,'weight_decay':1e-4,'grad_clip':5.,
                   'masked_weight':.2,'tail_weight':.02,'utility_weight':.2,'log_every':20,
                   'stage_epochs':{'evidence':1,'utility':1,'integration':1,'risk':1,'joint':1}},
       'evaluation':{'auto_evaluate':True},
       'analysis':{'enabled':True,'max_counterfactual_queries':1,'bootstrap_samples':20}}
    run_dir=tmp_path/'run'
    result=pipeline.run(c,torch.device('cpu'),run_dir=run_dir)
    assert (result/'best.pt').is_file()
    assert (result/'evaluation'/'performance_summary.csv').is_file()
    assert (result/'analysis'/'research_report.html').is_file()
    assert (result/'pipeline_complete.json').is_file()
    # Default post-training mode omits Val, but training still uses Val internally.
    import csv
    with open(result/'evaluation'/'performance_summary.csv',newline='') as fh:
        rows=list(csv.DictReader(fh))
    assert [r['split'] for r in rows]==['s20','s40','s60','s90','Avg_s20_s90','Avg_s20_s60']
    assert not (result/'evaluation'/'val').exists()
    assert not (result/'analysis'/'val').exists()
    assert (result/'analysis'/'counterfactual'/'candidate_counterfactual.csv').exists()
    import json
    findings=json.loads((result/'analysis'/'research_findings.json').read_text())
    assert findings['counterfactual_audit_split']=='s40'
