"""Normal-Anchored full-data TRAIN supervision and strictly masked inference."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import torch.nn.functional as F
import pytest
from test_v2_core import make_model,batch
from corsta_v2.data.dataset import structured_mask,_rng,KINDS,VALIDATION_KINDS
from corsta_v2.training.normal_anchor import normal_anchor_loss,actual_anchor_gain
from corsta_v2.counterfactual.teacher import PairedTeacher
from corsta_v2.training.trainer import stages_for,stage_optimizer

torch.set_num_threads(1)


def model_and_data(stage='anchor_cf'):
    model=make_model();model.prediction_mode='normal_anchored'
    model.set_stage(stage,use_graph=False)
    observed,b=batch()
    out=model(observed,enable_cf=stage!='normal',enable_controller=stage=='anchor_control')
    t=PairedTeacher(model.stla)(b,out['h0'],out['backbone'])
    config={'training':{'forecast_rmse_weight':0.05,'normal_residual_weight':0.1,
                        'cf_response_weight':0.06,'cf_error_gain_weight':0.03,
                        'cf_anchor_gain_weight':0.05,'cf_acceptance_weight':0.02,
                        'cf_error_temperature':0.35,'cf_error_rank_weight':0.02,
                        'cf_anchor_rank_weight':0.02,'cf_latent_weight':0.01}}
    return model,observed,b,out,t,config


def test_modes_keep_distinct_training_stages():
    assert stages_for({'model':{'prediction_mode':'normal_anchored'}})==('normal','anchor_cf','anchor_control')
    assert stages_for({'model':{'prediction_mode':'cf_first'}})==('cf','residual','joint','control')


def test_mask_types_and_unchanged_validation_patterns():
    assert 'entry_random' in KINDS
    assert VALIDATION_KINDS==('sensor','regional','directional','temporal')
    v=np.ones((12,9),dtype=bool);v[0,0]=False
    for kind in KINDS:
        m=structured_mask(v,40,kind,_rng(17,3,8,40,kind))
        assert not (m & ~v).any()
        assert v.sum()-m.sum()==round(v.sum()*.4)
    r=structured_mask(v,60,'entry_random',_rng(1,1,4,60,'entry_random'))
    assert np.any(r[0]) and np.any(~r[0])


def test_normal_teacher_forced_forecast_and_residual_backprop():
    m,obs,b,out,t,c=model_and_data('normal')
    loss,d=normal_anchor_loss(out,b,t,'normal',0,1,c)
    assert torch.isfinite(loss)
    assert 'normal_semantic' in d
    loss.backward()
    assert m.residual.head[-1].weight.grad is not None
    assert m.cf.shift[-1].weight.grad is None
    assert all(p.grad is None for p in m.stla.parameters())


def test_anchor_cf_freezes_entire_normal_anchor():
    m,obs,b,out,t,c=model_and_data('anchor_cf')
    original=out['normal_prediction'].detach().clone()
    frozen=[m.failure,m.residual,m.graph_residual,m.stla]
    assert all(not p.requires_grad for mod in frozen for p in mod.parameters())
    loss,d=normal_anchor_loss(out,b,t,'anchor_cf',0,1,c)
    loss.backward()
    assert m.cf.shift[-1].weight.grad is not None
    assert m.cf.anchor_gain_head[-1].weight.grad is not None
    assert m.cf.anchor_acceptance_head[-1].weight.grad is not None
    assert all(p.grad is None for mod in frozen for p in mod.parameters())
    opt=torch.optim.SGD([p for p in m.parameters() if p.requires_grad],lr=0.005)
    opt.step();m.train()
    after=m(obs,enable_cf=True,enable_controller=False)
    assert torch.allclose(after['normal_prediction'],original,atol=1e-7)


def test_anchor_fallbacks_and_cf_delta_reference():
    m,obs,b,out,t,c=model_and_data('anchor_cf');m.eval()
    with torch.no_grad():
        x=m(obs,enable_cf=False)
        assert torch.allclose(x['prediction'],x['normal_prediction'])
        assert torch.allclose(out['normal_cf_candidate'],out['normal_prediction']+out['delta_cf_candidate'])
        assert torch.allclose(out['delta_cf_candidate'],out['cf_raw_prediction']-out['backbone'])
        m.cf.anchor_acceptance_head[-1].bias.fill_(-25.)
        m.set_stage('anchor_control')
        suppressed=m(obs,enable_controller=True)
        assert torch.allclose(suppressed['prediction'],suppressed['normal_prediction'],atol=1e-6)
        m.cf.anchor_acceptance_head[-1].bias.fill_(25.)
        accepted=m(obs,enable_controller=True)
        assert torch.allclose(accepted['prediction'],accepted['normal_cf_candidate'],atol=1e-6)


def test_gain_is_candidate_relative_to_normal_not_backbone():
    y=torch.tensor([[0.,10.]])
    normal=torch.tensor([[1.,11.]])
    candidate=torch.tensor([[0.,14.]])
    gain=actual_anchor_gain(normal,candidate,y)
    assert torch.equal(gain,torch.tensor([[1.,-3.]]))


def test_anchor_multi_view_ranking_has_trainable_gradient():
    m,obs,b,out,t,c=model_and_data('anchor_cf')
    other={k:v.clone() for k,v in obs.items()}
    other['mask'][:,0]=0;other['x'][:,0]=0
    alt=m(other,enable_controller=False)
    loss,d=normal_anchor_loss(out,b,t,'anchor_cf',0,1,c,other_views=[alt])
    assert torch.isfinite(loss)
    assert 'anchor_mask_rank' in d and 'teacher_mask_rank' in d
    assert d['num_mask_views'].item()==2
    loss.backward()
    assert m.cf.anchor_gain_head[-1].weight.grad.abs().sum()>0


def test_control_trains_only_anchor_heads():
    m,obs,b,out,t,c=model_and_data('anchor_control')
    trainable=[name for name,p in m.named_parameters() if p.requires_grad]
    assert trainable and all(('anchor_gain_head' in n or 'anchor_acceptance_head' in n) for n in trainable)
    loss,d=normal_anchor_loss(out,b,t,'anchor_control',0,1,c)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.cf.anchor_acceptance_head[-1].weight.grad is not None
    assert m.residual.head[-1].weight.grad is None


def test_student_input_rejects_full_and_future():
    m,obs,b,out,t,c=model_and_data('anchor_cf')
    with pytest.raises(ValueError):m({**obs,'x_full':b['x_full']})
    with pytest.raises(ValueError):m({**obs,'y_norm':b['y_norm']})


def test_teacher_decomposition_for_normal_anchor():
    m,obs,b,out,t,c=model_and_data('anchor_cf')
    y=b['y_norm']
    assert torch.allclose(t['cf_target']+t['residual_target'],y-t['p_masked'],atol=2e-5)
    assert torch.allclose(t['cf_target'],t['p_full']-out['backbone'])


def test_time_embargoed_train_only_normal_calibration_partition():
    from corsta_v2.training.trainer import normal_calibration_partition
    class DatasetStub:
        window_starts=np.arange(400,dtype=np.int64)*12
    cfg={'training':{'normal_calibration_fraction':0.12,
                     'normal_calibration_embargo':24,'normal_calibration_limit':64}}
    part=normal_calibration_partition(DatasetStub(),cfg)
    assert part is not None
    training,heldout=part
    starts=DatasetStub.window_starts
    assert starts[training].max()+24<starts[heldout].min()
    assert len(heldout)==min(64,round(400*.12)) or len(heldout)==48
    assert set(training).isdisjoint(heldout)


def test_multiview_dataset_train_only_mask_and_shared_future():
    from integration_synthetic_v2 import synthetic_project
    from corsta_v2.data.dataset import CoRSTASplit
    import os
    root,_,_=synthetic_project('speed')
    cache=root/'fake_cache/speed/formal'
    ds=CoRSTASplit(cache,'train',seed=10,limit=4,num_views=4,
                   contrast_views=True,augment=True,
                   neighbors=[[] for _ in range(9)],directions=['N']*9)
    item=ds[0]
    assert item['view_x'].shape==(3,12,9)
    assert item['view_mask'].shape==(3,12,9)
    assert item['view_severity'].shape==(3,)
    assert item['view_kind'].shape==(3,)
    assert not torch.any(item['view_mask'].bool()&(~item['original_valid'].bool()).unsqueeze(0))
    for v in range(3):
        mask=item['view_mask'][v].bool()
        assert torch.allclose(item['view_x'][v][mask],item['x_full'][mask])
    ds.set_epoch(1)
    changed=ds[0]
    assert torch.equal(changed['y_norm'],item['y_norm'])
    assert torch.equal(changed['x_full'],item['x_full'])
    assert not torch.equal(changed['view_mask'],item['view_mask'])
    validation=CoRSTASplit(cache,'val',seed=10,limit=1)
    assert 'x_full' not in validation[0]
