#!/usr/bin/env python3
from pathlib import Path
import importlib.util
import numpy as np
import torch
import yaml

R=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('stateprobe',R/'scripts/run_predictive_need_state_probe.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
from presure.need_state import PredictiveNeedStateNet, binary_metrics, confusion_matrix_np, macro_f1_from_cm

cfg=yaml.safe_load((R/'configs/presure.yaml').read_text())
B,N,P,F=3,5,12,72
base_start=48
x=torch.randn(B,N,F)
net=PredictiveNeedStateNet(F,N,P,base_start,2,hidden=32,demb=16,node_dim=8,horizon_dim=4,dropout=0.0)
logits=net(x);assert logits.shape==(B,P,N,2)
y=torch.randint(0,2,(B,P,N));loss=torch.nn.functional.cross_entropy(logits.reshape(-1,2),y.reshape(-1));loss.backward();assert torch.isfinite(loss)

# Train-only threshold/label contract using synthetic horizon-specific values.
thresholds={'q50':[1.0]*P,'q75':[2.0]*P,'q80':[2.5]*P,'q90':[3.0]*P,'magnitude_band_centers_raw':[[.5,1.7,3.4] for _ in range(P)]}
err=np.zeros((B,P,N),np.float32);err[...,0]=4.0;err[...,1]=-3.5;err[...,2]=2.2;err[...,3]=1.5;err[...,4]=0.2
valid=np.ones_like(err,bool)
for task in ['need_q75','need_q90','direction_q75','direction_q90','magnitude_3','signed_state_q75']:
    yy,use=probe.make_task_target(task,err,valid,thresholds,cfg);assert yy.shape==err.shape and use.shape==err.shape
assert probe.make_task_target('need_q75',err,valid,thresholds,cfg)[0][0,0].tolist()==[1,1,1,0,0]
assert probe.make_task_target('signed_state_q75',err,valid,thresholds,cfg)[0][0,0].tolist()==[1,2,1,0,0]

m=binary_metrics(np.array([.1,.2,.8,.9]),np.array([0,0,1,1]));assert m['auroc']>.99 and m['f1']>.99
cm=confusion_matrix_np([0,1,2,2],[0,1,2,1],3);assert macro_f1_from_cm(cm)>0.6

# Decision contract.
tasks={
 'need_q75':{'validation':{'masked':{'auroc':.68}}},
 'need_q90':{'validation':{'masked':{'auroc':.70}}},
 'direction_q75':{'validation':{'masked':{'balanced_accuracy':.63}}},
 'direction_q90':{'validation':{'masked':{'balanced_accuracy':.64}}},
}
util={'M3_CONF':{'pairwise_rank_accuracy':.58},'PREDICTED_STATE_ANALYTIC':{'pairwise_rank_accuracy':.66},'TRUE_STATE_ANALYTIC':{'pairwise_rank_accuracy':.85}}
status,diag=probe.state_decision(tasks,util,cfg);assert status=='GO_V08_STATE' and diag['need_state_signal_pass']
print('PRESURE V0.7.2 PREDICTIVE NEED STATE PROBE SMOKE PASS',status)
