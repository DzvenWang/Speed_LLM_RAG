import importlib.util
from pathlib import Path
import numpy as np
import torch

def load(p):
    sp=importlib.util.spec_from_file_location('tested',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
C=load(Path(__file__).resolve().parents[1]/'tcrr_stla'/'diagnostics_v3'/'common.py')

def make_data():
    y=np.array([[[1.,2.],[3.,4.]],[[2.,4.],[6.,8.]]],dtype=np.float32)
    valid=np.ones_like(y,dtype=bool)
    direct=np.array([[1,0],[0,1]],dtype=bool)
    base=np.zeros_like(y)
    return y,valid,direct,base

def test_protocol_digest_batch_independent():
    keys=C.ProtocolDigest.FIELDS
    data={'window_start':torch.tensor([10,20,30]),'x':torch.randn(3,12,4),'mask':torch.ones(3,12,4),
          'tod':torch.zeros(3,12,dtype=torch.int64),'dow':torch.ones(3,12,dtype=torch.int64),
          'y_raw':torch.randn(3,12,4),'future_valid':torch.ones(3,12,4,dtype=torch.bool),'direct_mask':torch.ones(3,4,dtype=torch.bool)}
    a=C.ProtocolDigest();a.update(data)
    b=C.ProtocolDigest();b.update({k:v[:1] for k,v in data.items()});b.update({k:v[1:] for k,v in data.items()})
    assert a.result()==b.result()

def test_masked_metric_aggregation():
    y,v,d,base=make_data()
    # exact correction one query, leave second unchanged
    p=base.copy();p[0]=y[0]
    stats=C._query_stats(y,v,d,base,p)
    assert stats['count'].tolist()==[4,4]
    assert np.isclose(stats['full_l1'][0],0)
    assert np.isclose(stats['full_l1'][1],20)
    assert np.isclose(stats['direct_count'][0],2)
    group=C.summary([{'severity':20,'mode':'full',**{k:float(val[i]) for k,val in stats.items()}} for i in range(2)],['severity','mode'])
    assert np.isclose(group['MAE'].iloc[0],20/8)
    assert np.isclose(group['MAE_gain'].iloc[0],10/8)

def test_query_horizon_and_save_resume(tmp_path):
    y,v,d,base=make_data();b={'window_start':torch.tensor([10,20]),'index':torch.arange(2)}
    rows=[];h=[]
    C.add_mode(rows,b,base,y,v,d,base,'full',20,{'family':['a','b']})
    C.add_horizon(h,b,base,y,v,d,base,20,family=['a','b'])
    assert len(rows)==2 and len(h)==4
    C.save_severity(tmp_path/'s20',rows,h,[{'severity':20,'test':1}],{'status':'PASS'})
    assert (tmp_path/'s20'/'family_horizon.csv').is_file()
