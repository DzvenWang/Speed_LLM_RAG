from pathlib import Path
import sys,tempfile
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from experiment_common.constants import *
from experiment_common.time_features import resolve_time_features,assert_resolved_time_features
from experiment_common.metrics import metric_bundle
from experiment_common.predictions import save_prediction_npz,load_prediction_npz
class B:
    def __init__(self,root,T=576):
        self.traffic=np.zeros((T,2,1)); self.cfg={'forecast':{'freq_minutes':5}}; self.sg_dir=Path(root)
with tempfile.TemporaryDirectory() as td:
    p=Path(td); (p/'core').mkdir(); T=576
    pd.DataFrame({'index':np.arange(T),'timestamp':pd.date_range('2026-01-05',periods=T,freq='5min')}).to_csv(p/'core/time_index.csv',index=False)
    r=resolve_time_features(B(p,T)); a=assert_resolved_time_features(r)
    assert a['tod_unique']==288 and a['dow_unique']>=2 and r.parser=='datetime_string'
    pred=np.zeros((2,12,3),np.float32); y=np.ones_like(pred); valid=np.ones_like(pred,np.uint8); node=np.ones((2,3),np.uint8)
    m=metric_bundle(pred,y,valid,node,node); assert m['overall_MAE']==1.0 and m['overall_RMSE']==1.0
    save_prediction_npz(p/'pred.npz',['a','b'],pred); vids,pp=load_prediction_npz(p/'pred.npz',expected_rows=2,expected_horizon=12,expected_nodes=3); assert list(vids)==['a','b'] and pp.shape==(2,12,3)
assert CONTRACT_VERSION=='1.0.0'; assert DATA_CONTRACT_REVISION=='traffident_experiment_cache_v1'
print('SHARED EXPERIMENT CONTRACT SMOKE PASS')
