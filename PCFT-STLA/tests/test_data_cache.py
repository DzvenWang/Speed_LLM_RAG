import json,sys
from pathlib import Path
import numpy as np,pandas as pd
import pytest
from pcft_stla.v6.data import make_multi_cache,MultiTargetSplit

class Bench:
    def __init__(self,root):
        self.sg_dir=root;self.H=12;self.P=12;self.N=4
        self.channel_names=['flow','occupancy','speed'];self.target_name='speed'
        self.manifest={'primary_channel':'speed','version':'1'}
        self.stats={'channels':{c:{'mean':0.,'std':1.} for c in self.channel_names}}
        self.traffic=np.ones((40,4,3),np.float32)
        self.valid=np.ones((40,4,3),bool)
        self.valid[0,0,0]=False  # invalidate flow not speed; mask must target flow
        self.views=pd.DataFrame({'view_id':['a','b']})
    def get_view_spec(self,view_id):return {'view_id':view_id,'window_start':0 if view_id=='a' else 1,'split':'train'}
    def normalize(self,a):return a.copy()

def test_multivariate_cache_flow_target_and_masks(tmp_path,monkeypatch):
    import supportbench.sensing as sense
    # Mask is selected by FLOW target valid mask, not benchmark primary SPEED.
    def fake_mask(spec,H,N,sg,valid_target):
        a=np.zeros((H,N),bool)
        a[0,1]=valid_target[0,1]
        a[0,0]=valid_target[0,0]
        return a
    monkeypatch.setattr(sense,'materialize_mask',fake_mask)
    b=Bench(tmp_path)
    target_root=tmp_path/'target';target_root.mkdir()
    meta=pd.DataFrame({'view_id':['a','b'],'window_start':[0,1]})
    meta.to_csv(target_root/'train_manifest.csv',index=False)
    json.dump({'target_name':'flow'},open(target_root/'cache_manifest.json','w'))
    expected=[];mask_expected=[]
    for st in [0,1]:
        v=b.valid[st:st+12];art=fake_mask({},12,4,tmp_path,v[...,0]);o=v[...,0]&~art
        expected.append(np.where(o,b.traffic[st:st+12,:,0],0.))
        mask_expected.append(o.astype('uint8'))
    def save(name,x):np.save(target_root/('train_'+name+'.npy'),x)
    save('x_obs_norm',np.array(expected,dtype='float32'))
    save('history_observed',np.array(mask_expected,dtype='uint8'))
    save('y_raw',np.ones((2,12,4),np.float32));save('y_norm',np.ones((2,12,4),np.float32))
    save('future_valid',np.ones((2,12,4),np.uint8));save('direct_mask',np.ones((2,4),np.uint8))
    for name in ['tod_hist','dow_hist','tod_future','dow_future']:
        save(name,np.zeros((2,12),np.int64))
    out=make_multi_cache(b,target_root,'train',tmp_path/'multi')
    ds=MultiTargetSplit(target_root,'train',out)
    sample=ds[0]
    assert sample['x_multi'].shape==(12,4,3)
    assert sample['observed_mask'][0,0,0]==0   # originally invalid Flow
    assert sample['observed_mask'][0,1,2]==0   # artificial shared mask hides Speed too
    assert sample['x_target'][0,1]==0
    assert len(ds)==2
    # Reopening aligned cache is permitted; corrupt identity is not.
    assert make_multi_cache(b,target_root,'train',tmp_path/'multi')==out
    np.save(tmp_path/'multi'/'x.npy',np.zeros((1,12,4,3),np.float32))
    with pytest.raises(RuntimeError):make_multi_cache(b,target_root,'train',tmp_path/'multi')
