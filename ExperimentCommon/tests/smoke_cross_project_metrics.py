from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT/'ExperimentCommon', ROOT/'Backbone', ROOT/'Baseline'):
    sys.path.insert(0,str(p))
from experiment_common.metrics import metric_bundle, backbone_metric_summary
from baseline.metrics import metric_bundle as baseline_metric_bundle
from backbone.metrics import backbone_metric_summary as backbone_summary

rng=np.random.default_rng(10)
pred=rng.normal(size=(4,12,5)).astype(np.float32)
true=rng.normal(size=(4,12,5)).astype(np.float32)
valid=np.ones_like(pred,dtype=np.uint8)
direct=np.array([[1,0,1,0,0],[0,1,1,0,0],[1,1,0,0,1],[0,0,1,1,0]],dtype=np.uint8)
support=np.array([[1,1,0,0,0],[0,1,0,1,0],[1,0,0,0,1],[0,0,1,0,1]],dtype=np.uint8)
a=metric_bundle(pred,true,valid,direct,support)
b=baseline_metric_bundle(pred,true,valid,direct,support)
assert a==b
c=backbone_metric_summary(pred,true,valid,direct,support)
d=backbone_summary(pred,true,valid,direct,support)
assert c==d
assert np.isclose(c['MAE'],a['overall_MAE'])
assert np.isclose(c['directly_masked_MAE'],a['masked_MAE'])
assert np.isclose(c['support_dependent_MAE'],a['support_MAE'])
print('CROSS-PROJECT METRICS CONTRACT SMOKE PASS')
